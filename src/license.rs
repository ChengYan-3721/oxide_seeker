use crate::{
    error::{AppError, Result},
    storage::database::{self, DbPool, LicenseRow},
};
use chrono::{Local, TimeZone, Utc};
use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use serde::{Deserialize, Serialize};

const TRIAL_DAYS: i64 = 30;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LicenseClaims {
    pub sub: String,
    pub exp: usize,
    pub iat: usize,
    pub is_permanent: bool,
    pub machine_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LicenseStatusResponse {
   pub status: String, // trial | valid | permanent | expired | invalid
   pub message: String,
   pub trial_days_total: i64,
   pub trial_days_used: i64,
   pub trial_days_left: i64,
   pub expires_at: Option<String>,
   pub customer: Option<String>,
   pub machine_id: String,
   pub search_allowed: bool,
}

#[derive(Debug, Clone)]
pub struct LicenseState {
    pub status: String,
    pub message: String,
    pub trial_days_used: i64,
    pub trial_days_left: i64,
    pub expires_at: Option<i64>,
    pub customer: Option<String>,
    pub search_allowed: bool,
}

pub fn get_machine_id() -> String {
    machine_uid::get().unwrap_or_else(|e| {
        tracing::warn!("获取机器码失败: {}", e);
        "UNKNOWN-MACHINE-ID".to_string()
    })
}

pub async fn evaluate_and_persist(pool: &DbPool) -> Result<LicenseState> {
    let machine_id = get_machine_id();
    let row = database::get_or_init_license_row(pool, &machine_id).await?;
    evaluate_row_and_persist(pool, row, &machine_id).await
}

pub async fn save_license_key_and_evaluate(pool: &DbPool, license_key: &str) -> Result<LicenseState> {
    let machine_id = get_machine_id();
    let maybe = verify_license_key(license_key, &machine_id);
    let (claims, message) = match maybe {
        Some(v) => v,
        None => {
            return Err(AppError::InvalidRequest("许可无效或已过期".to_string()));
        }
    };

    database::update_license_key(pool, Some(license_key)).await?;

    let expires_at = if claims.is_permanent {
        None
    } else {
        Some(claims.exp as i64)
    };
    let status = if claims.is_permanent { "permanent" } else { "valid" };
    let validated_at = Some(Utc::now().timestamp());
    database::update_license_state(
        pool,
        status,
        Some(&message),
        expires_at,
        Some(&claims.sub),
        validated_at,
    )
    .await?;

    Ok(LicenseState {
        status: status.to_string(),
        message,
        trial_days_used: 0,
        trial_days_left: 0,
        expires_at,
        customer: Some(claims.sub),
        search_allowed: true,
    })
}

pub async fn clear_license_key_and_evaluate(pool: &DbPool) -> Result<LicenseState> {
    database::update_license_key(pool, None).await?;
    evaluate_and_persist(pool).await
}

pub async fn current_status(pool: &DbPool) -> Result<LicenseStatusResponse> {
    let state = evaluate_and_persist(pool).await?;
    Ok(to_response(&state))
}

pub fn to_response(state: &LicenseState) -> LicenseStatusResponse {
   LicenseStatusResponse {
       status: state.status.clone(),
       message: state.message.clone(),
       trial_days_total: TRIAL_DAYS,
       trial_days_used: state.trial_days_used,
       trial_days_left: state.trial_days_left,
       expires_at: state
           .expires_at
           .and_then(|ts| Local.timestamp_opt(ts, 0).single())
           .map(|dt| dt.to_rfc3339()),
       customer: state.customer.clone(),
       machine_id: get_machine_id(),
       search_allowed: state.search_allowed,
   }
}

async fn evaluate_row_and_persist(pool: &DbPool, row: LicenseRow, machine_id: &str) -> Result<LicenseState> {
    if row.install_fingerprint.trim().is_empty() {
        return Err(AppError::Config("install_fingerprint 初始化失败".to_string()));
    }

    if let Some(ref key) = row.license_key {
        if !key.trim().is_empty() {
            if let Some((claims, message)) = verify_license_key(key, machine_id) {
                let expires_at = if claims.is_permanent {
                    None
                } else {
                    Some(claims.exp as i64)
                };
                let status = if claims.is_permanent { "permanent" } else { "valid" };
                let validated_at = Some(Utc::now().timestamp());
                database::update_license_state(
                    pool,
                    status,
                    Some(&message),
                    expires_at,
                    Some(&claims.sub),
                    validated_at,
                )
                .await?;

                return Ok(LicenseState {
                    status: status.to_string(),
                    message,
                    trial_days_used: 0,
                    trial_days_left: 0,
                    expires_at,
                    customer: Some(claims.sub),
                    search_allowed: true,
                });
            }
        }
    }

    let now = Utc::now().timestamp();
    let started = row.install_started_at;
    let passed_days = ((now - started).max(0)) / 86_400;
    if passed_days < TRIAL_DAYS {
        let left = TRIAL_DAYS - passed_days;
        let msg = format!("试用中：已使用 {} 天，剩余 {} 天", passed_days, left);
        database::update_license_state(pool, "trial", Some(&msg), None, None, None).await?;
        return Ok(LicenseState {
            status: "trial".to_string(),
            message: msg,
            trial_days_used: passed_days,
            trial_days_left: left,
            expires_at: None,
            customer: None,
            search_allowed: true,
        });
    }

    let msg = "许可失效：试用期已结束，请输入有效许可".to_string();
    database::update_license_state(pool, "expired", Some(&msg), None, None, None).await?;
    Ok(LicenseState {
        status: "expired".to_string(),
        message: msg,
        trial_days_used: TRIAL_DAYS,
        trial_days_left: 0,
        expires_at: None,
        customer: None,
        search_allowed: false,
    })
}

fn verify_license_key(license_key: &str, machine_id: &str) -> Option<(LicenseClaims, String)> {
    const PUBLIC_KEY_PEM: &[u8] = include_bytes!("../public.pem");
    let decoding_key = DecodingKey::from_ed_pem(PUBLIC_KEY_PEM).ok()?;

    let mut validation = Validation::new(Algorithm::EdDSA);
    validation.validate_exp = true;
    validation.leeway = 60;

    match decode::<LicenseClaims>(license_key, &decoding_key, &validation) {
        Ok(token_data) => {
            let claims = token_data.claims;
            if claims.machine_id != machine_id {
                tracing::warn!(
                    "许可机器码不匹配，license={}, current={}",
                    claims.machine_id,
                    machine_id
                );
                return None;
            }

            let msg = if claims.is_permanent {
                format!("许可有效：{}（永久）", claims.sub)
            } else {
                let exp_text = Local
                    .timestamp_opt(claims.exp as i64, 0)
                    .single()
                    .map(|d| d.to_rfc3339())
                    .unwrap_or_else(|| claims.exp.to_string());
                format!("许可有效：{}（有效期至 {}）", claims.sub, exp_text)
            };
            Some((claims, msg))
        }
        Err(e) => {
            tracing::warn!("许可校验失败: {}", e);
            None
        }
    }
}