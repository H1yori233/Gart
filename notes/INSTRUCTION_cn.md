# 项目3 CUDA路径追踪器 - 说明

提交截止日期为**10月1日**晚上11:59。

这个项目需要相当长的运行时间来生成高质量图像,请将这一点考虑在内。你将获得额外2天时间(截止到10月3日)仅用于"README和场景"的更新。但是,10月1日的标准项目要求对README仍然适用。你可以利用这两天额外的时间来改进图像、图表、性能分析等。

如果你计划在这个项目上使用延期天数(我们建议这样做),可以将其应用于代码截止日期(这也会推迟README截止日期),或者仅应用于README截止日期。例如,你可以使用一个延期天将代码截止日期推迟到10月2日,README截止日期推迟到10月4日,然后再使用另一个延期天将README截止日期进一步推迟到10月5日,总共使用两个延期天。

[链接到"路径追踪入门"幻灯片](https://docs.google.com/presentation/d/1pQU_qkxx9Pq9h2Y20tLvE7v7AwaA_6byvszXi9Y-K7A/edit?usp=drive_link)

**概述:**

在这个项目中,你将实现一个基于CUDA的路径追踪器,能够快速渲染全局光照图像。由于在本课程中我们关注的是GPU编程、性能和生成实际的精美图像(而不是诸如I/O之类的平凡编程任务),因此本项目包含了用于加载场景描述文件的基础代码(如下所述),以及其他一些通常构成预览和保存图像框架的内容。

核心渲染器留给你来实现。最后请注意,虽然这个基础代码旨在作为CUDA路径追踪器的有力起点,但如果你不想使用它,你也不是必须使用它。你可以随意更改基础代码的任何部分,甚至从头开始。**这是你的项目。**

**建议:**

* 你保存的每张图像都应该自动获得不同的文件名。不要删除它们!为了你的README的利益,保留一些图像,这样你就可以在最后选择一些来记录你的进展。非常欢迎出现意外的效果!
* 记得保存你的调试图像 - 这些将成为一个很好的README。
* 也要记得保存和分享你的失误。每张图像都有一个故事要讲,我们想听听。

## 内容

* `src/` C++/CUDA源文件。
* `scenes/` 示例场景描述JSON文件。
* `img/` 示例场景描述文件的渲染结果。(这些可能与你的不完全匹配。)
* `external/` 第三方库的包含文件和静态库。

## 运行代码

主函数需要一个场景描述文件。用一个场景文件作为参数调用程序:`cis565_path_tracer scenes/sphere.json`。(在Visual Studio中,使用`../scenes/sphere.json`。)

如果你使用Visual Studio,可以在`项目属性`的`调试 > 命令参数`部分设置这个。确保路径正确 - 查看控制台是否有错误。

### 控制

* Esc键保存图像并退出。
* S键保存图像。注意控制台输出的文件名。
* 空格键将相机重新居中到原始场景的观察点。
* 左鼠标按钮旋转相机。
* 右鼠标按钮在垂直轴上进行缩放。
* 中键移动场景X/Z平面中的观察点。

## 要求

在这个项目中,你获得了以下代码:

* 加载和读取场景描述格式的代码。
* 球体和盒子的相交函数。
* 支持保存图像。
* 用于在运行时预览渲染的工作CUDA-GL互操作。
* 一个骨架渲染器,包含:
  * 简单的射线-场景相交。
  * 一个"假"着色内核,根据材质和相交属性为射线着色,但不基于BSDF计算新的射线。

**如有疑问请在Ed Discussion中提问。**

### 第1部分 - 核心功能

你需要实现以下功能:

* 带有BSDF评估的着色内核,用于:
  * 理想漫反射表面(使用提供的余弦加权散射函数,见下文。) [PBRTv4 9.2](https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection)
  * 完全镜面反射(镜面)表面(例如使用`glm::reflect`)。
  * 参见`scatterRay`中关于漫反射/镜面反射的说明,以及下面关于不完美镜面反射的说明。
* 使用项目2中的流压缩进行路径延续/终止。
* 在你有了[基本的路径追踪器](img/REFERENCE_cornell.5000samp.png)之后,
  实现一种使射线/路径段/相交点按材质类型在内存中连续的方法。这应该可以轻松切换。
  * 考虑在缓冲区中为每个路径段着色并使用一个大的着色内核执行BSDF评估的问题:内核中不同的材质/BSDF评估将需要不同的时间来完成。
  * 在着色之前对射线/路径段进行排序,使得与相同材质交互的射线/路径在内存中连续。这如何影响性能?为什么?
* 最后,实现随机采样抗锯齿。参见Paul Bourke的[笔记](https://paulbourke.net/miscellaneous/raytracing/)中的"随机采样"部分。



### 第2部分 - 让你的路径追踪器与众不同!

以下功能是一个非详尽的列表,你可以根据自己的兴趣和动力从中选择。每个功能都有一个相关的分数(用表情数字表示,例如:five:)。

**你需要从下面的列表中选择实现总计至少10分的额外功能。**

一个可选功能的示例集是:

* 网格加载 - :four: 分
* 折射 - :two: 分
* 景深 - :two: 分
* 最终射线后处理 - :three: 分

这个列表并不完整。如果你有特别想实现的想法(例如加速结构等),请在Ed上发帖。

**额外加分**: 在上述必需功能之上实现更多功能,根据难度和创意,评分者可酌情给予最多+25/100的加分。

#### 视觉改进

* :two: 折射(例如玻璃/水)[PBRTv4 9.3](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission),使用[Schlick近似](https://en.wikipedia.org/wiki/Schlick's_approximation)或更准确的方法[PBRTv4 9.5](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF.html)实现菲涅耳效应。你可以使用`glm::refract`实现斯涅尔定律。
  * 推荐但不要求:非完美镜面表面(见下文"不完美镜面照明")。
* :two: 基于物理的景深(通过在光圈内抖动射线)。[PBRTv4 5.2.3](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models#TheThinLensModelandDepthofField)
* :four: 程序化形状和纹理。
  * 你必须程序化生成至少两种不同的复杂形状。(不是基本图元)
  * 你必须能够用至少两种不同的纹理为对象着色
* :five: (:six: 如果与任意网格加载结合)纹理映射[PBRTv4 10.4](https://pbr-book.org/4ed/Textures_and_Materials/Image_Texture)和凹凸映射[PBRTv3 9.3](https://www.pbr-book.org/3ed-2018/Materials/Bump_Mapping.html)。
  * 实现文件加载纹理和基本程序化纹理
  * 提供两者之间的性能比较
* :two: 直接照明(通过将最终射线直接指向作为光源的发光物体上的随机点)。或更高级的[PBRTv4 13.4](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer)。
* :four: 次表面散射[PBRTv3 5.6.2](https://www.pbr-book.org/3ed-2018/Color_and_Radiometry/Surface_Reflection#TheBSSRDF), [11.4](https://www.pbr-book.org/3ed-2018/Volume_Scattering/The_BSSRDF.html)。
* :three: [用于蒙特卡洛光线追踪的更好的随机数序列](https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_07_Random.pdf)
* :three: 某种定义物体运动的方法,以及通过平均动画中不同时间的样本来实现运动模糊。
* :three: 使用最终射线应用后处理着色器。请在开始之前在Piazza上发布你的想法。

#### 网格改进

* 任意网格加载和渲染(例如glTF 2.0(首选)或`obj`文件),可切换包围体相交剔除
  * :four: glTF
  * :two: OBJ
  * 对于其他格式,请在课程论坛上查询
  * 你可以在网上找到模型或从你喜欢的3D建模应用程序导出。经批准后,你可以使用第三方加载代码将数据导入C++。
    * 强烈推荐使用[tinygltf](https://github.com/syoyo/tinygltf/)用于glTF。
    * 强烈推荐使用[tinyObj](https://github.com/syoyo/tinyobjloader)用于OBJ。
    * [obj2gltf](https://github.com/CesiumGS/obj2gltf)可用于将OBJ转换为glTF文件。你可以找到类似的项目用于FBX和其他格式。
  * 你可以使用三角形相交函数`glm::intersectRayTriangle`。
  * 包围体相交剔除:通过首先检查射线与完全包围网格的体积的相交来减少需要检查整个网格的射线数量。要获得全部学分,请提供有无此优化的性能分析。
  
  > 注意:这与层次空间数据结构搭配得很好。

#### 性能改进

* :one: 实现俄罗斯轮盘赌路径终止,它可以在不引入偏差的情况下提前终止不重要的路径。确保包括启用和禁用它的性能评估,特别是对于封闭场景。[PBRTv3 13.7](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting)
* :two: 使用跨多个块的共享内存进行工作效率流压缩。(参见[*GPU Gems 3*第39章](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)。)
  * 注意,如果你在项目2中将共享内存流压缩作为额外学分实现,则不会为此获得额外学分。
* :six: 层次空间数据结构 - 用于更好的射线/场景相交测试
  * 推荐BVH或八叉树 - 这个功能更多是关于GPU上的遍历而不是完美的树结构
  * CPU端数据结构构建就足够了 - GPU端构建是一个[期末项目](https://github.com/jeremynewlin/Accel)
  * 确保这是可切换的以进行性能比较
  * 如果与任意网格加载(本年度要求)结合实现,这可以作为可切换的包围体相交剔除。
  * 更多资源见下文
* :six: [波前路径追踪](https://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-gpus):
无需排序即可按材质对射线进行分组。一个合理的实现将需要相当大的重构,因为每个支持的材质突然需要自己的内核。
* :three: [*Open Image AI Denoiser或替代批准的图像去噪器*](https://github.com/OpenImageDenoise/oidn) Open Image Denoiser是一个图像去噪器,它通过对基于蒙特卡洛的路径追踪器输出应用滤波器来工作。去噪器在CPU上运行,接受从1spp到更高的路径追踪器输出。为了获得全部学分,你必须至少传入一个额外的缓冲区以及[原始"美感"缓冲区](https://github.com/OpenImageDenoise/oidn#open-image-denoise-overview)。**例如:** 美感 + 法线。
  * 这个额外学分的部分是要弄清楚应该在哪里调用滤波器,以及如何管理滤波步骤的数据。
  * 重要的是要注意,集成这个并不像一开始看起来那么简单。库集成、缓冲区创建、设备兼容性等都是会出现的真实问题,调试它们可能很困难。请只有在提前完成第2部分并想要额外分数时才尝试这个。虽然这很困难,但结果将是路径追踪图像的显著更快的分辨率。
* :five: 可重启路径追踪:保存一些应用程序状态(迭代次数、到目前为止的样本、加速结构),这样你就可以开始和停止渲染,而不是让你的计算机运行数小时(这在这个项目中会发生)
* :five: 将项目从使用CUDA-OpenGL互操作切换到使用CUDA-Vulkan互操作(对于那些对使用Vulkan感兴趣的人来说,这是一个很好的选择)。如果你计划追求这个,请与助教交谈。

#### 优化

**对于那些对渲染主题不太感兴趣的人**,我们鼓励你专注于使用GPU编程技术和更高级的CUDA功能来优化基本路径追踪器。
除了核心功能外,我们确实建议至少实现一个OBJ网格加载器,然后再专注于优化,这样你就可以加载重型几何体来开始看到性能影响。
请参考课程材料(特别是CUDA性能讲座)和[CUDA最佳实践指南](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)了解如何优化CUDA性能。
一些例子包括:
* 使用共享内存来提高内存带宽
* 使用内部函数来提高指令吞吐量
* 使用CUDA流和/或图进行并发内核执行

对于每种特定的优化技术,请在Ed Discussion上发帖,以便我们确定适当的分数奖励。

## 分析

对于每个额外功能,你必须提供以下分析:

* 功能的概述写作以及前/后图像。
* 功能的性能影响。
* 如果你做了什么来加速该功能,你做了什么以及为什么?
* 将你的GPU版本的功能与假设的CPU版本进行比较(你不必实现它!)。它是从在GPU上实现中受益还是受损?
* 这个功能如何在你当前的实现之外进行优化?

## 基础代码导览

你将在以下文件中工作。寻找代码的重要部分:

* 搜索`CHECKITOUT`。
* 你必须实现标有`TODO`的部分。(但不要让这些限制你 - 你有完全的自由!)

* `src/pathtrace.h`/`cu`: 路径追踪内核、设备函数和调用代码
  * `pathtraceInit`初始化路径追踪器状态 - 它应该从`Scene`复制场景数据(例如几何体、材质)。
  * `pathtraceFree`释放由`pathtraceInit`分配的内存
  * `pathtrace`执行渲染的一次迭代 - 它处理内核启动、内存复制、传输一些数据等。
    * 参见注释以了解低级路径追踪回顾。
* `src/intersections.h`/`cu`: 射线相交函数
  * `boxIntersectionTest`和`sphereIntersectionTest`,它们接受一个射线和一个几何对象,并返回相交的各种属性。
* `src/interactions.h`/`cu`: 射线散射函数
  * `calculateRandomDirectionInHemisphere`: 半球中的余弦加权随机方向。实现漫反射表面所需。
  * `scatterRay`: 这个函数应该执行所有射线散射,并将调用`calculateRandomDirectionInHemisphere`。详见注释。
* `src/main.cpp`: 你不需要在这里做任何事情,但如果你想的话,你可以更改程序以保存`.hdr`图像文件(用于后处理)。
* `stream_compaction`: 一个虚拟文件夹,你应该将项目2中的流压缩实现放在其中。复制[这里](https://github.com/CIS5650-Fall-2024/Project2-Stream-Compaction/tree/main/stream_compaction)的文件应该就足够了。

### 生成随机数

```cpp
thrust::default_random_engine rng(hash(index));
thrust::uniform_real_distribution<float> u01(0, 1);
float result = u01(rng);
```

有一个便利函数用于使用索引、迭代和深度的组合作为种子生成随机引擎:

```cpp
thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, path.remainingBounces);
```

### 不完美镜面照明

在路径追踪中,像漫反射材质一样,镜面材质是使用概率分布来模拟的,而不是基于角度计算射线反弹的强度。

[*GPU Gems 3*第20章](https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling)的公式7、8和9给出了生成随机镜面射线的公式。(注意有一个印刷错误:文本中的χ = 公式中的ξ。)

另请参见`scatterRay`中关于漫反射/镜面/其他材质类型之间概率分配的注释。

另见:[PBRTv3 8.2.2](https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#SpecularReflection)。

### 层次空间数据结构

避免检查射线与场景中每个基本体或网格中每个三角形的一种方法是将基本体放入层次空间数据结构中,如[八叉树](https://en.wikipedia.org/wiki/Octree)。

射线-基本体相交然后涉及递归地测试射线与树的不同层级中的包围体的相交,直到达到包含基本体/三角形子集的叶子,此时检查射线与叶子中所有基本体/三角形的相交。

* 我们强烈建议在CPU上构建数据结构,并将树缓冲区封装到它们自己的结构中,并有自己专用的GPU内存管理函数。
* 我们强烈建议在编写任何实际代码之前手动处理几个案例的树构建算法。
  * 该算法如何分配空间中均匀分布的三角形?
  * 如果模型是一个完美的轴对齐立方体,有6个面12个三角形会怎样?这个测试经常会带来许多边缘情况!
* 注意在GPU上的遍历必须以迭代方式编码!
* 在GPU上良好的执行需要调整最大树深度。从一开始就使这个可配置。
* 如果一个基本体跨越数据结构中多个叶子单元,对于这个项目来说,在每个叶子单元中计算基本体就足够了。

### 处理长时间运行的CUDA线程

默认情况下,如果CUDA内核运行超过5秒,你的GPU驱动程序可能会终止它。有一种方法可以禁用这个超时。只是要注意无限循环 - 它们可能会锁定你的计算机。

> 对于Cuda编程,禁用TDR最简单的方法是,假设你已安装NVIDIA Nsight工具,打开Nsight Monitor,点击"Nsight Monitor选项",在"常规"下将"WDDM TDR enabled"设置为false。这将为你更改注册表设置。关闭并重启。对TDR注册表设置的任何更改在重启之前都不会生效。[Stack Overflow](http://stackoverflow.com/questions/497685/cuda-apps-time-out-fail-after-several-seconds-how-to-work-around-this)

### 场景文件格式

> 注意:场景文件格式和示例场景文件是作为起点提供的。我们鼓励你创建自己独特的场景文件,甚至完全修改场景文件格式。请务必在你的readme中记录任何更改。

这个项目使用基于JSON的场景描述格式来定义场景的所有组件,如材质、对象、灯光和相机设置。场景文件被构造为一个JSON对象,对不同元素进行清晰的组织,提供了一个干净且可扩展的格式。

### 材质

材质在"Materials"部分下定义。每个材质都有一个唯一的名称,并属于一个材质类型,如"Diffuse"、"Specular"或"Emitting"。

对于每种类型的材质,它可以有不同的属性,如:

- "RGB": 一个包含三个浮点值的数组,定义材质的颜色。
- "EMITTANCE": 发光材质的浮点值,定义光发射强度(可选,仅存在于发光材质)。
- "ROUGHNESS": 表示表面粗糙度的浮点值,用于镜面材质。

示例:

```
"diffuse_red": {
    "TYPE": "Diffuse",
    "RGB": [0.85, 0.35, 0.35]
}
```

### 相机

相机配置在"Camera"部分定义。它包括输出图像的分辨率、视场角、渲染迭代次数和相机方向等设置。

- "RES": 表示输出图像分辨率(以像素为单位)的数组。
- "FOVY": 垂直视场角,以度为单位。
- "ITERATIONS": 渲染过程中细化图像的迭代次数。
- "DEPTH": 最大路径追踪深度。
- "FILE": 渲染输出的文件名。
- "EYE": 相机在世界坐标中的位置。
- "LOOKAT": 相机指向的空间点。
- "UP": 定义相机方向的上向量。

示例:

```
"Camera": {
    "RES": [800, 800],
    "FOVY": 45.0,
    "ITERATIONS": 5000,
    "DEPTH": 8,
    "FILE": "cornell",
    "EYE": [0.0, 5.0, 10.5],
    "LOOKAT": [0.0, 5.0, 0.0],
    "UP": [0.0, 1.0, 0.0]
}
```

### 对象

场景中的对象在"Objects"部分下定义为一个条目数组。每个对象包含:

- "TYPE": 对象类型,如"cube"或"sphere"。
- "MATERIAL": 分配给对象的材质,引用前面定义的材质之一。
- "TRANS": 对象的平移(位置)数组。
- "ROTAT": 对象的旋转数组,以度为单位。
- "SCALE": 对象的缩放数组。

示例:

```
{
    "TYPE": "cube",
    "MATERIAL": "diffuse_red",
    "TRANS": [-5.0, 5.0, 0.0],
    "ROTAT": [0.0, 0.0, 0.0],
    "SCALE": [0.01, 10.0, 10.0]
}
```

这个JSON格式是灵活的,可以轻松扩展以适应新功能,如额外的材质属性或对象类型。

## 第三方代码政策

* 使用任何第三方代码必须通过在我们的Ed Discussion上询问获得批准。
* 如果获得批准,所有学生都可以使用它。通常,我们批准使用不是项目核心部分的第三方代码。例如,对于路径追踪器,我们会批准使用第三方库来加载模型,但不会批准复制和粘贴用于折射的CUDA函数。
* 第三方代码**必须**在README.md中注明出处。
* 未经批准使用第三方代码,包括使用其他学生的代码,是学术诚信违规行为,至少会导致你在本学期获得F。
* 你可以在项目中使用第三方3D模型和场景。请务必按照创作者的要求提供适当的归属。

## README

请参见:[**编写精彩README的技巧**](https://github.com/pjcozzi/Articles/blob/master/CIS565/GitHubRepo/README.md)

* 推销你的项目。
* 假设读者对路径追踪有一些了解 - 不要详细解释它是什么。专注于你的项目。
* 不要把它说成是一个作业 - 不要说什么是"额外"或"额外学分"。谈论你完成了什么。

* 使用它来记录你做了什么。
* 你的封面图片*不应该*是康奈尔盒子 - 展示一些更有趣的东西!
  * 如果你要大幅定制它,请通过Ed Discussion预先获得批准。
* *不要*把README留到最后一刻!
  * 它是项目的关键部分,如果没有一个好的README,我们将无法给你评分。
  * 生成图像需要时间。一定要考虑到这一点!

此外:

* 这是一个渲染器,所以要包含你制作的图像!
* 确保用数字和比较来支持你的优化声明。
* 如果你引用任何其他材料,请提供链接。
* 你不会因为路径追踪器运行的速度而被评分,但接近实时总是很好!
* 如果你有一个快速的GPU渲染器,用视频展示交互性是很好的。如果这样做,请包含链接!

### 分析

* 流压缩在几次反弹后帮助最大。打印并绘制单次迭代内流压缩的效果(即每次反弹后未终止射线的数量),并评估你从流压缩获得的好处。
* 比较开放的场景(如给定的康奈尔盒)和封闭的场景(即光线无法逃离场景)。再次比较流压缩的性能效果!记住,流压缩只影响终止的射线,所以你可能会期待什么?
* 对于针对特定内核的优化,我们建议使用堆叠条形图来传达总执行时间和单个内核的改进。例如:

  ![显然Macchiato是最优的。](img/stacked_bar_graph.png)

  NSight的计时应该对生成这类图表非常有用。

## 提交

如果你修改了任何`CMakeLists.txt`文件(除了`SOURCE_FILES`列表之外),请明确提及。

注意Piazza上讨论的任何构建问题。

打开一个GitHub拉取请求,这样我们就可以看到你已经完成了。

标题应该是"Project 3: 你的名字"。

你的拉取请求评论部分的模板如下,你可以复制粘贴:

* [仓库链接](链接到你的仓库)
* (简要)提及你已完成的功能。特别是那些你想要突出的额外功能
  * 功能0
  * 功能1
  * ...
* 对项目本身的反馈(如果有的话)。

## 参考资料

* [PBRTv3] [基于物理的渲染:从理论到实现 (pbr-book.org)](https://www.pbr-book.org/3ed-2018/contents)
* [PBRTv4] [基于物理的渲染:从理论到实现 (pbr-book.org)](https://pbr-book.org/4ed/contents)
* 抗锯齿和光线追踪。Chris Cooksey和Paul Bourke, https://paulbourke.net/miscellaneous/raytracing/
* Steve Rotenberg和Matteo Mannino的[采样笔记](http://graphics.ucsd.edu/courses/cse168_s14/), 加利福尼亚大学圣地亚哥分校, CSE168:渲染算法
* 路径追踪器README示例(非详尽列表):
  * https://github.com/byumjin/Project3-CUDA-Path-Tracer
  * https://github.com/lukedan/Project3-CUDA-Path-Tracer
  * https://github.com/botforge/CUDA-Path-Tracer
  * https://github.com/taylornelms15/Project3-CUDA-Path-Tracer
  * https://github.com/emily-vo/cuda-pathtrace
  * https://github.com/ascn/toki
  * https://github.com/gracelgilbert/Project3-CUDA-Path-Tracer
  * https://github.com/vasumahesh1/Project3-CUDA-Path-Tracer
