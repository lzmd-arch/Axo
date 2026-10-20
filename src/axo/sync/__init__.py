"""桌面与 Android 之间的记忆同步。

`protocol` / `devices` / `audit` 是纯层；`certs` / `service` / `server` 才碰
文件、TLS 与 socket。同步只允许"读"与"追加"，不存在 update/delete。
"""
