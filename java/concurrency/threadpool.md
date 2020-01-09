# 线程池

## 为什么要使用线程池

池化技术主要是为了减少每次获取资源的时间，提高对资源的利用率。

使用线程池游有以下好处：

1. 通过重复利用已经创建的资源降低资源消耗。
2. 提高速度（不需要创建资源就可以执行）。
3. 提高线程的可管理性。

# Excutor 框架

## 简介

Excutor 是继 Java5 之后引进的，通过 Excutor 来启动线程比使用 Thread 的 start 方法更优化，除了易于管理，效率高（使用线程池，节约开销) 外，还可以避免 this 逃逸问题。

this 逃逸是指在构造函数返回之前就持有该对象的引用，调用尚未狗仔完全的方法可能引发的错误。

Excutor 框架提供了线程池的管理，线程工厂、队列以及拒绝策略等等，让并发编程变得更简单。

### Excutor 框架结构

* **任务 Runnable/Callable**

  执行任务需要实现 `Runnable` 或`Callable`接口，这两个接口的实现类都可以被 `ThreadPoolExecutor`或者 `ScheduledThreadExecutor` 执行。

* **任务的执行 Exucutor**
* **异步计算的结果 Future** 

### Excutor 使用流程

1. **主线程首先要创建实现 `Runnable` 或者 `Callable` 接口的任务对象。**
2. **把创建完成的实现 `Runnable`/`Callable`接口的 对象直接交给 `ExecutorService` 执行**: `ExecutorService.execute（Runnable command）`）或者也可以把 `Runnable` 对象或`Callable` 对象提交给 `ExecutorService` 执行（`ExecutorService.submit（Runnable task）`或 `ExecutorService.submit（Callable  task）`）。
3. **如果执行 `ExecutorService.submit（…）`，`ExecutorService` 将返回一个实现`Future`接口的对象**（我们刚刚也提到过了执行 `execute()`方法和 `submit()`方法的区别，`submit()`会返回一个 `FutureTask 对象）。由于 FutureTask` 实现了 `Runnable`，我们也可以创建 `FutureTask`，然后直接交给 `ExecutorService` 执行。
4. **最后，主线程可以执行 `FutureTask.get()`方法来等待任务执行完成。主线程也可以执行 `FutureTask.cancel（boolean mayInterruptIfRunning）`来取消此任务的执行。**

