## MySQL知识总结

### 什么是MySQL？

MySQL 是一种关系型数据库。因为其开源免费，并且方便扩展，在Java企业级开发中非常常用。任何人都可以在GPL(General Public License) 的许可下下载并根据个性化的需要对其进行修改。 MySQL的默认端口是3306。

### 存储引擎

**常用命令**

* 查看MySQL提供的左右存储引擎

```mysql
mysql> show engines;
```

MySQL 当前默认的存储引擎是InnoDB, 并且在5.7版本所有的存储引擎中只有InnoDB是事务性存储引擎，也就是说只有InnoDB支持事物。

