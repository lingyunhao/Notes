## Git 版本管理系统

### Remote and Local repository

Remote repo: 专门服务器，可多人共享

Local repo：个人机器上使用

### Git 的三个工作区和文件的三种状态

我们在使用Git的时候一般有三个工作区的概念：**Git 仓库、工作目录以及暂存区域**。工作目录是我们开发时修改文件的那些目录，Git仓库是项目目录下面的.git目录中的 `.git` 目录中的内容，而暂存区域是存放已经被Git标记过，将要提交保存到Git数据库的文件的地方。

文件可能处于三种状态其一：

1. Modified(已修改): 已经修改了文件，但是还没保存到数据库中。
2. Staged(已暂存): 表示对一个已修改文件的当前版本做了标记，使之包含在下次提交的快照中。
3. Committed(已提交): 数据已安全地保存在本地数据库中。

* 刚开始编辑文件，文件处于**modified**状态，是在**工作目录**下。
* 修改完文件，使用`git add`，文件将变成为**staged**状态，文件进入**暂存区域**，内容将被保存在Git数据库中。
* 当我们执行`git commit`，文件成为**commited**状态，创建了一个提交记录保存到了git仓库中。

**暂存区域**

暂存区域是一个文件，保存了下次要提交的文件信息列表，一般在Git仓库目录中，对应于**index文件**中的内容。我们可以通过`git ls-files --stage`来查看里面的内容。

`100644 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0    Git.md`

暂存区里记录了文件的内容所对应的数据对应以及文件路径。实际上，暂存区域里保存的是系列指向文件的索引，真正的文件内容都在`.git/objects`目录下，也就是Git数据库中。

**我们执行的Git操作，几乎z只往数据库中增加数据，不会删除数据**

## Git 使用

### 创建数据库

1. 创建全新的repo，在现有目录中初始化仓库: 进入项目目录运行 `git init`命令，该命令将创建一个名为`.git`的子目录。
2. 从一个服务器克隆一个现有的repo: `git clone [url]` 自定义本地repo的名字 `git clone [url] directoryname`

### 修改文件并更新到本地repo

1. 检测当前文件状态: `git status`
2. 把处于modified状态的文件添加到缓存区：
   * 特定文件：`git add filename`
   * 所有文件：`git add *`
   * 所有.txt文件：`git add *.txt`

3. 忽略文件： `.gitignore filename`
4. 提交更新：`git commit -m "info about this commit"` (commit 之前要先`git status`看下是否所有需要commit的文件都保存到了暂存区域)
5. 跳过使用暂存区域更新的方式：`git commit -a -m "info about this commit"`  `git commit`加上 `-a`就会把所有已经跟踪过的文件暂存起来一并提交，从而跳过`git add`步骤。
6. 移除文件：`git rm filename` 先从暂存区域移除，然后提交（首先要在本地 rm移除掉已经被git track的文件，后面详细说明）
7. 对文件重命名: `git mv README.md README` 相当于`mv README.md README`、`git rm README.md` 、`git add README` 这三条命令的集合。

