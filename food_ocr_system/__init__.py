import pymysql

# 用纯 Python 的 pymysql 驱动代替 MySQLdb（无需编译 mysqlclient）
pymysql.install_as_MySQLdb()
