#  Learning Differentiable Logic Programs for Abstract Visual Reasoning
Hikaru Shindo, Viktor Pfanschilling, Devendra Singh Dhami, Kristian Kersting

<!-- ![neumann](./imgs/neumann_logo_mid_large.png) -->

<p align="center">
  <img src="./imgs/neumann_logo_mid_large.png">
</p>

# Abstract
Visual reasoning is essential for building intelligent agents that understand the world and perform problem-solving beyond perception. Differentiable forward reasoning has been developed to integrate reasoning with gradient-based machine learning paradigms. 
However, due to the memory intensity, most existing approaches do not bring the best of the expressivity of first-order logic, excluding a crucial ability to solve *abstract visual reasoning*, where agents need to perform reasoning by using analogies on abstract concepts in different scenarios. 
To overcome this problem, we propose *NEUro-symbolic Message-pAssiNg reasoNer (NEUMANN)*, which is a graph-based differentiable forward reasoner, passing messages in a memory-efficient manner and handling structured programs with functors.
Moreover, we propose a computationally-efficient structure learning algorithm to perform explanatory program induction on complex visual scenes.
To evaluate, in addition to conventional visual reasoning tasks, we propose a new task, *visual reasoning behind-the-scenes*, where agents need to learn abstract programs and then answer queries by imagining scenes that are not observed.
We empirically demonstrate that NEUMANN solves visual reasoning tasks efficiently, outperforming neural, symbolic, and neuro-symbolic baselines.


![neumann](./imgs/behind-the-scenes.png)

**NEUMANN solves Behind-the-Scenes task.**
Reasoning behind the scenes:  The goal of this task is to compute the answer of a query, e.g., *``What is the color of the second left-most object after deleting a gray object?''* given a visual scene. To answer this query, the agent needs to reason behind the scenes and understand abstract operations on objects. In the first task, the agent needs to induce an explicit program given visual examples, where each example consists of several visual scenes that describe the input and the output of the operation to be learned. The abstract operations can be described and computed by first-order logic with functors. 
In the second task, the agent needs to apply the learned programs to new situations to solve queries reasoning about non-observational scenes.

# Relevant Repositories
[Visual ILP: A repository of the dataset generation of CLEVR images for abstract operations.](https://github.com/ml-research/visual-ilp)

[Behind-the-Scenes: A repository for the generation of visual scenes and queries for the behind-the-scenes task.](https://github.com/ml-research/behind-the-scenes)


# LICENSE
See [LICENSE](./LICENSE). The [src/yolov5](./src/yolov5) folder is following [GPL3](./src/yolov5/LICENSE) license.