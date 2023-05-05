---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---


# Abstract

DiSProD is an online planner developed for environments with probabilistic transitions in continuous state and action spaces. DiSProD builds a symbolic graph that captures the distribution of future trajectories, conditioned on a given policy, using independence assumptions and approximate propagation of distributions. The symbolic graph provides a differentiable representation of the policy's value, enabling efficient gradient-based optimization for long-horizon search. The propagation of approximate distributions can be seen as an aggregation of many trajectories, making it well-suited for dealing with sparse rewards and stochastic environments. An extensive experimental evaluation compares DiSProD to state-of-the-art planners in discrete-time planning and real-time control of robotic systems. The proposed method improves over existing planners in handling stochastic environments, sensitivity to search depth, sparsity of rewards, and large action spaces. Additional real-world experiments demonstrate that DiSProD can control ground vehicles and surface vessels to successfully navigate around obstacles. 
{: class='abstract-text'}

# Experiments

### Controlling Jackal
<div class="experiment-videos">
<iframe width="560" height="315" src="https://www.youtube.com/embed/3YjPtmSiHr0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/dtjz-oIN7Nk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

### Controlling Heron

<div class="experiment-videos">
<iframe width="560" height="315" src="https://www.youtube.com/embed/o_tq4TFZWqQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/foT2mMUbJfU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/gXDCNo-lvr8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>