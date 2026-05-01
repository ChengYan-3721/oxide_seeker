fn main() {
    if cfg!(target_os = "windows") {
        let mut res = winres::WindowsResource::new();
        res.set_icon("logo.ico");
        // 嵌入UAC清单，请求管理员权限
        res.set_manifest(r#"
             <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
               <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
                 <security>
                   <requestedPrivileges>
                     <requestedExecutionLevel level="requireAdministrator" uiAccess="false" />
                   </requestedPrivileges>
                 </security>
               </trustInfo>
             </assembly>
             "#);
        res.compile().unwrap();
    }
}
