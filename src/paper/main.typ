//==============================================================================
// ifacconf.typ 2023-11-17 Alexander Von Moll
// Template for IFAC meeting papers
//
// Adapted from ifacconf.tex by Juan a. de la Puente
//==============================================================================

#import "@preview/abiding-ifacconf:0.1.0": *
#show: ifacconf-rules
#show: ifacconf.with(
  title: "RimWorld Combat Agent: 
  A Reinforcement Learning Approach",
  authors: (
    (
      name: "Linzheng Tang (2022533087)",
      email: "tanglzh2022@shanghaitech.edu.cn",
      affiliation: 1,
    ),
    (
      name: "Guo Yu (2022533074)",
      email: "yuguo2022@shanghaitech.edu.cn",
      affiliation: 1,
    ),
    (
      name: "Junlin Chen (2022533009)",
      email: "chenjl2022@shanghaitech.edu.cn",
      affiliation: 1,
    ),
  ),
  affiliations: (
    (
      organization: "ShanghaiTech University",
    ),
  ),  abstract: [
    This paper presents agents implemented by several different reinforcement learning approaches, regarding the combat mechanism in the single player game RimWorld. 
    The agents are trained to develop strategies to defeat the enemy in the one-on-one shooting scenario, based on Deep-Q-learning, Proximal Policy Optimization and Policy Gradient algorithms.
  ],
  keywords: ("RimWorld", "Deep-Q-learning", "Proximal Policy Optimization", "Policy Gradient"),

)

= Introduction

== Game Background
RimWorld is a construction and management simulation video game developed by Canadian game designer Tynan Sylvester and published by Ludeon Studios. 


== Combat Scenario

=== Map
Specifically, the agents are applied to a 2 dimensional map of 15x15 grids. A complex architectural structure is placed in the middle and trees are uniformly distributed, walls forming the structure and trees are fixed. 

The walls are impenetrable, hiding behind them and the trees can reduce the other party's shooting accuracy towards the pawn. 


=== Pawns
The enemy pawn and ally pawn are born diagonally opposite to each other with the same equipments, health status and skill points. The ally pawn can go anywhere in the map where the point is not blocked by walls, while the enemy pawn will wander around birth point in defense. 

Once the two pawns are in each other's sight, they will exchange fire automatically with aiming intervals between shots, the aiming process will be interrupted if the pawn moves during it.

=== Winning Criteria
When a pawn suffers too much blood loss, pain or certain critical wounds, it will be paralyzed and unable to continue fighting (not necessarily dead). Under this circumstance, the game is deemed over and the other party will be declared as the winner. If both parties are paralyzed at the same time, ally wins.  

The detailed explanation of the paralyzing mechanism will be covered in the design of the reward function in the following sections.

=== Other Factors
Weather conditions and time slots are unmodified and will preserve certain randomness of the game. For instance, shooting accuracy of both parties will be reduced in foggy rain or at night.

== Combat Map

#figure(
  image("images/paper/map.png", height: 50%, width: 80%,fit: "contain"),
  caption: [Combat Map],
) <map>


= Environment
The training environment is built based on Gymnasium, which is a python reinforcement learning framework maintained by Farama Foundation. The reinforcement learning agents are applied in the form of a game mod, named "Combat Agent". 

The server side(agent) and client side(game) has a communication interval of 0.5 seconds, agents are trained in parallel in the vectorized environment with the following specifications:

== Action Space
Due to the short communication interval, the action space is restricted by the pawn's walking speed. 
The action space is therefore defined as a unit length nine-square grid, centering the ally pawn, whose relative coordinates is set to (0, 0).

An action is defined as the coordinate within the relative nine-square grid.

== Observation Space
The observation space consists of 6 layers in total, each layer represents certain information of the current game state and is shaped 15x15.
Those layers are viewed as 6 channels of the observation space.

=== Ally position layer
A binary map, where the value is 1 if the ally pawn is at the corresponding position, otherwise 0.

=== Enemy position layer
A binary map, where the value is 1 if the enemy pawn is at the corresponding position, otherwise 0.

=== Cover positions layer
A discrete map, where 1 represents the tree position and 2 represents the wall position.

=== Aiming layer
A binary map, where the value is 1 if the ally pawn or the enemy pawn is aiming, otherwise 0. This layer
helps the agent determine whether to interrupt the aiming process of both parties.

=== Status layer
A binary map, where the value is 1 if the ally or enemy pawn is paralyzed, otherwise 0. 

=== Danger layer
A discrete map, where the value is 0-100, representing the danger level of both parties. The specific calculation
of danger value will be elaborated in the reward function section.

=== Frame Stack
The 6 layers present the current game state, however, in actual combating, the agent needs information through the timeline to make a better decision.

Therefore, the observation space is stacked per 8 frames, making the observation space 6x15x15x8, those frames are viewed as the depth dimension.

#figure(
  image("images/paper/obs.jpg", height: 16%, width: 80%, fit: "stretch"),
  caption: [Single Frame Observation Space Channels],
) <obs>

== Episode Termination
The episode is terminated if one of the following conditions is met:
1. A winner is declared.
2. The maximum step length 800 is reached. In which case, episode reward will be preserved with no change.

== Reward Function
After each episode is terminated, the episode reward will be calculated and recorded based on the following reward table:

#tablefig(
  table(
    columns: 3,
    align: center + horizon,
    stroke: none,
    inset: 3pt,
    [Reward Type], [Enemy], [Ally],
    table.hline(),
    [Original], [/], [0],
    [Win], [-50], [+50],
    [Danger], [+200], [-200],
    [Invalid Action], [/], [-0.25],
    [Distance], [/], [±0.1],
    [Cover], [/], [+0.15],
    table.hline(),
  ),
  caption: [Reward Settings],
) <reward_settings>

Notice that only the currently in use rewards are given in the table, other rewards such as ally remain still penalty are designed but deprecated, their coefficients are set to 0.

=== Win
If a winner is declared and it is the ally, the ally will be rewarded 50 points, otherwise it will be penalized 50 points.


=== Danger  
The original paralyzing mechanism in the game is calculated through a linear piecewise function regarding the pawn's consciousness, pain and blood loss stage. 

In actual calculations, to unify the computing system, the paralyzing danger degree is defined as follows:

$
d = 1 / (1 + exp(5 + c - p - b))
$ <danger_formula>

where $c$ is the pawn's consciousness proportion, $p$ is the pawn's pain proportion and $b$ is the pawn's blood loss proportion, they are
floats ranging from 0 to 1. 

The danger value is mapped to (0, 1) with 2 decimal places by sigmoid function. When the value is greater than 0.75, the pawn will be paralyzed. The corresponding reward is calculated by the difference equation:

$
r_d = c_d dot.c (d_(n+1) - d_n)
$ <danger_reward>

where $r_d$ is the danger reward , $c_d$ is the danger reward coefficient, $d_(n+1)$ is the danger value of the current step, $d_n$ is the danger value of the previous step.

Notice that the paralyzing danger degree is calculated separately from the game's 
internal paralyzing mechanism, and when passed to the Danger layer, it is 
mapped to an integer within (0, 100).

=== Invalid Action
When the ally pawn tries to move to a position that is blocked by walls or outside its action space, the action is deemed invalid.

Invalid coordinates will be clipped to the action space and reward will be penalized $c_i$ points, where $c_i$ is the invalid action coefficient.

=== Distance
The ally pawn tends to perform better when it keeps certain distance from the enemy pawn. In practice, the optimal distance is set to 5 with a slack 
range of 1 grid. Once the ally is out of the optimal distance range, the reward will be penalized $c_d$ points, otherwise the reward will be added $c_d$ points.

=== Cover
The ally is encouraged to take cover behind walls or trees and hide from the enemy's straight shot. The action space of ally is divided into north-east, north-west, south-east and south-west quadrants, the quadrant closest to the enemy is activated, every cover within such quadrant will be rewarded $c_c$ points, where $c_c$ is the cover reward coefficient.


= Deep-Q-learning
<deep-q-learning>
== Neural Network
<neural-network>
=== Double Q-learning
<double-q-learning>
The DQN agent uses double Q-learning to stabilize the training process.
The agent consists of two identical Q-networks: the policy network and
the target network. 

The policy network is used to estimate the Q-value
of the current state, while the target network is used to estimate the
Q-value of the next state. This separation helps mitigate the
overestimation bias inherent in traditional Q-learning. The Q-value of
the current state is calculated using the Bellman equation:

$ Q lr((s_t comma a_t)) eq r_t plus gamma dot.op max_(a_(t plus 1)) Q lr((s_(t plus 1) comma a_(t plus 1))) comma $

where $s_t$ is the current state, $a_t$ is the action taken, $r_t$ is
the reward received, $gamma$ is the discount factor, and $s_(t plus 1)$
is the next state. The target network is periodically updated to match
the policy network, ensuring stable and consistent Q-value estimates.

=== Distributional Q-learning
<distributional-q-learning>
Distributional Q-learning extends traditional Q-learning by modeling the
distribution of possible returns rather than just their expected values.
Instead of estimating a single Q-value for each state-action pair, the
agent predicts a probability distribution over a discrete set of
possible returns (atoms). 

This approach captures the inherent
uncertainty in the environment and provides richer information for
decision-making. The distributional Q-value is computed as:

$ Z lr((s_t comma a_t)) eq sum_(i eq 1)^N p_i dot.op delta_(z_i) comma $

where $p_i$ is the probability of the $i$-th atom $z_i$, and
$delta_(z_i)$ is the Dirac delta function centered at $z_i$. The
expected Q-value can then be derived by taking the weighted sum of the
atoms:

$ Q lr((s_t comma a_t)) eq sum_(i eq 1)^N p_i dot.op z_i dot.basic $

This method improves the agent’s ability to handle stochastic
environments and leads to more robust learning.

=== ResNet Block
<resnet-block>
The first block of the model is based on the ResNet architecture,
specifically the R3D-18 variant, which is adapted for 3D input data. The
ResNet block consists of a series of convolutional layers with residual
connections, allowing the network to learn deeper representations
without suffering from vanishing gradients. 

The input is passed through
a stem convolutional layer, followed by multiple residual blocks. Each
residual block includes a combination of 3D convolutions, batch
normalization (replaced with group normalization for stability), and
ReLU activation functions. The output of the ResNet block is a
high-dimensional feature representation of the input state, which is
then fed into the subsequent layers of the network.

=== Noisy Linear Layer
<noisy-linear-layer>
The noisy linear layer is a variant of the standard linear layer that
introduces noise to the weights and biases during training. This noise
encourages exploration by adding uncertainty to the network’s
predictions, which is particularly useful in reinforcement learning
tasks where exploration is critical. The noisy linear layer is defined
as:

$ y eq lr((W_mu plus W_sigma dot.op epsilon.alt_W)) dot.op x plus lr((b_mu plus b_sigma dot.op epsilon.alt_b)) comma $

where $W_mu$ and $b_mu$ are the mean weights and biases, $W_sigma$ and
$b_sigma$ are the standard deviations of the noise, and $epsilon.alt_W$
and $epsilon.alt_b$ are random noise terms sampled from a standard
normal distribution. 

During evaluation, the noise is disabled, and the
layer behaves like a standard linear layer. This approach allows the
agent to balance exploration and exploitation effectively.

=== Dueling Network Block
<dueling-network-block>
The dueling network architecture separates the estimation of state
values and action advantages, which are then combined to produce the
final Q-value estimates. This separation allows the network to learn
which states are valuable without needing to learn the effect of each
action in every state. 

The dueling network consists of two streams: the
value stream and the advantage stream. The value stream estimates the
value of the state $V lr((s))$, while the advantage stream estimates the
advantage of each action $A lr((s comma a))$. The two streams are
combined using the following formula:

$ Q lr((s comma a)) eq V lr((s)) plus lr((A lr((s comma a)) minus 1 / lr(|A|) sum_(a prime) A lr((s comma a prime)))) comma $

where $lr(|A|)$ is the number of actions. This ensures that the Q-values
are centered around the state value, improving the stability of the
learning process. 



























Both streams consist of multiple noisy linear layers
with ReLU activations, followed by a final fully connected layer that
outputs the value and advantage estimates. The final Q-value
distribution is obtained by applying a softmax function to the combined
output.


#figure(
  image("images/paper/dqn.jpg", fit: "stretch"),
  caption: [DQN Model Structure],
) <dqn_model>

== Action

=== Epsilon Greedy
The DQN agent uses epsilon greedy strategy. Specifically, the epsilon decay follows the 
following formula:

$
epsilon = max(epsilon_("final"), epsilon_0 times (1 - exp(-5 times d^s)))
$ <epsilon_formula>

where $d$ is the decay rate, $epsilon_0$ is the initial epsilon, $epsilon_("final")$ is the final epsilon threshold, $s$ is the step number. In practice, $epsilon_0$ is set to 1.0, $epsilon_("final")$ is set to 0.01, $d$ is set to 0.9999935.

The agent will take the action with the highest Q value with probability $1 - epsilon$ and a random action sampled from the action space with probability $epsilon$.

== Update
<update>

=== Prioritized Experience Replay
<prioritized-experience-replay>
The DQN agent employs prioritized experience replay to improve sample efficiency and learning stability. Instead of uniformly sampling experiences from the replay buffer, the agent prioritizes experiences based on their temporal difference (TD) errors. 

Experiences with higher TD errors are more likely to be sampled, as they provide more significant learning signals. The priority of each experience is computed as:

$ p_i eq |delta_i| plus epsilon_p comma $

where $delta_i$ is the TD error for the $i$-th experience, and $epsilon_p$ is a small constant to ensure that all experiences have a non-zero probability of being sampled. The sampling probability for each experience is then given by:

$ P(i) eq p_i^alpha / sum_j p_j^alpha comma $

where $alpha$ controls the degree of prioritization. When $alpha = 0$, the sampling becomes uniform, and when $alpha = 1$, the sampling is fully prioritized. To correct for the bias introduced by prioritized sampling, importance sampling weights are applied during the update:

$ w_i eq lr((N dot.op P(i)))^(-beta) comma $

where $N$ is the size of the replay buffer, and $beta$ controls the degree of importance sampling correction. The weights are normalized to ensure stability during training.

=== Policy Network Update
<policy-network-update>
The policy network is updated using the sampled experiences and their corresponding importance weights. The loss function for the policy network is based on the KL divergence between the predicted Q-value distribution and the target distribution. The target distribution is computed using the target network and the Bellman equation:

$ T lr((s_t comma a_t)) eq r_t plus gamma dot.op Z lr((s_(t plus 1) comma a_(t plus 1))) comma $

where $Z lr((s_(t plus 1) comma a_(t plus 1)))$ is the distributional Q-value predicted by the target network for the next state-action pair. The loss function is then given by:

$ L eq sum_i w_i dot.op "KL" lr((Q lr((s_t^i comma a_t^i)) comma T lr((s_t^i comma a_t^i)))) comma $

where $Q lr((s_t^i comma a_t^i))$ is the predicted Q-value distribution for the $i$-th experience, and $"KL"$ is the Kullback-Leibler divergence. The policy network is updated by minimizing this loss using gradient descent.

=== Target Network Update
<target-network-update>
The target network is periodically updated to match the policy network, ensuring stable and consistent Q-value estimates. This update is performed by copying the weights of the policy network to the target network at regular intervals. 

The update frequency is controlled by a hyperparameter, typically set to a few thousand steps. This periodic update helps mitigate the overestimation bias inherent in traditional Q-learning and stabilizes the training process.

=== Priority Update
<priority-update>
After each policy network update, the priorities of the sampled experiences are updated based on the new TD errors. The updated priorities are computed as:

$ p_i^"new" eq |delta_i^"new"| plus epsilon_p comma $

where $delta_i^"new"$ is the new TD error for the $i$-th experience. These updated priorities are then stored in the replay buffer, ensuring that the sampling probabilities reflect the most recent learning signals. 

This dynamic prioritization allows the agent to focus on the most informative experiences, leading to more efficient learning.



== Training Results
The DQN agent is trained with 1,000,000 steps, the recorded
inner parameter curve is shown in the following figure:

#figure(
  image("images/paper/dqn_loss.png", fit: "contain"),
  caption: [Loss Curve],
) <loss_curve>

The detailed training reward and actual tactical development of the agent will 
be shown in the result comparison section.




= Proximal Policy Optimization
The general workflow of PPO is the same as DQN, the difference is that the update of PPO is performed with a fixed interval with temporal 
memory, which makes it generally faster in convergence than DQN.


== Neural Network 

=== Convolution block
The structure of PPO convolution block is the same as the DQN convolution block mentioned in the previous section.

=== Actor block
The actor block is responsible for the action sampling of the agent, which consists of two Linear ReLU layers and a full connected layer.

The output of the actor block is then scaled and used to calculate the action mean value and action standard deviation.

=== Critic block
The critic block is responsible for the state value estimation, which consists of two Linear ReLU layers and a full connected layer.

The output state value will be used for advantage calculation in the update step.


#figure(
  image("images/paper/ppo.jpg", fit: "contain"),
  caption: [PPO Model Structure],
) <ppo_model>


== Action  

=== Sampling
After forward propagation, the next action will be sampled from a certain distribution with action log probability. 

In such process, the nine-square action space is mapped to a 9x1 action list,
ranging from 0 to 8, where each number represents one of the nine positions in the action space.

For both x and y coordinates, the next x and y will be sampled and rounded separately from the following Categorical distribution, which is widely used in describing the discrete action space:
$
"Mapped"(x, y) ~ "Categorical"(l)
$

where $l$ is the log probability of the action, $x$ and $y$ are coordinates of the action.

=== Evaluation
In this part, the state tensor will go through forward propagation again, the action will be resampled from the same distribution.
The log probability and distribution entropy of the action will be used for the calculation of loss in the update step.

== Update
The update step is performed once every 500 steps in practice, key concept is using the temporal difference and clipped 
advantage to calculate total loss and use back propagation for parameter update.


=== GAE (Generalized Advantage Estimation) 
This method is used to calculate the advantage of the current state,
the calculation flow is as follows:


Temporal difference calculation:
$
delta_t = r_t + gamma V(s_(t+1)) dot.c (1 - "done"_t) - V(s_t)
$ <delta_formula>
where $r_t$ is the reward of the current state, $gamma$ is the discount factor, which is set to 0.97 in practice, $V(s_(t+1))$ is the state value of the next state, $"done"_t$ is the termination flag of the current state.

Current state advantage calculation:
$
A_t = delta_t + gamma lambda (1 - "done"_t) A_(t+1)
$ <advantage_formula>
where $lambda$ is the GAE parameter, which is set to 0.95 in practice, $A_(t+1)$ is the advantage of the next state.

Current state return calculation:
$
R_t = A_t + V(s_t)
$ <returns_formula>
where $R_t$ is the return of the current state.

Normalized advantage calculation:
$
A^hat_t = A_t - mu_A / (sigma_A + epsilon)
$ <normalized_advantage>
where $mu_A$ is the mean value of the advantage, $sigma_A$ is the standard deviation of the advantage, $epsilon$ is a small number to prevent division by zero,

which is set to $10^(-8)$ in practice.


=== Actor Loss
The actor loss is associated with the estimated advantage and entropy, the calculation follows equation:

Surrogate loss 1:
$
  s_1 = exp(l_t - l_(t-1)) dot.c A^hat_t
$
where $l_t$ is the log probability of the current state, $l_(t-1)$ is the log probability of the previous state, $A^hat_t$ is the normalized advantage of the current state.

Surrogate loss 2:
$
  s_2 = "clamp"(exp(l_t - l_(t-1)), 1 - c, 1 + c) dot.c A^hat_t
$
where $c$ is the clip parameter, which is set to 0.1 in practice, clamp is a function that clips the value to the range of (1 - c, 1 + c).

Notice that the two surrogate losses are calculated using absolute values instead of exponential functions to prevent value explosion.

Entropy bonus:
$
e_b = e_c times "entropy"(s_t)
$
where $e_b$ is the entropy bonus, $e_c$ is the entropy coefficient, entropy($s_t$) is the entropy of the current state,
in practice, $e_c$ is set to 0.05.


Total Actor loss:
$
L_a = -min(s_1, s_2) - e_b
$
where $L_a$ is the actor loss.

=== Critic Loss
The critic loss is associated with the state value, the calculation follows equation:

$
L_c = (R_t - V(s_t))^2
$
where $L_c$ is the critic loss, $R_t$ is the return of the current state, $V(s_t)$ is the state value of the current state.

=== Total Loss
The total loss is the sum of the actor loss and the critic loss, the calculation follows equation:

$
L = L_a + L_c times c_c
$
where $L$ is the total loss, $c_c$ is the critic coefficient.




== Training Results

The PPO agent is trained with 1,000,000 steps, the recorded
loss curve is shown in the following figure:

#figure(
  image("images/paper/ppo_loss.png", fit: "stretch"),
  caption: [Loss Curve],
) <loss_curve>

The detailed training reward and actual tactical development of the agent will 
be shown in the result comparison section.








= Policy Gradient Method

== Neural Network

=== Convolution Block
The structure of the convolution block is the same as the DQN convolution block mentioned in the previous section.

=== Actor block
Similar to the PPO actor block mentioned in the previous section, the actor block is responsible for the action sampling of the agent, which consists of two Linear ReLU layers and a full connected layer.

#figure(
  image("images/paper/pgm.jpg", fit: "stretch"),
  caption: [PGM Model Structure],
) <pgm_model>

== Action


=== Sampling
Similar to PPO, the exact action coordinates will be sampled from two separate Categorical distributions and mapped to the action space.

=== Evaluation
The same as PPO except that the action would not be forwarded the second time,
only log probability and entropy of the action will be recorded as the sum of the two coordinates' log probability and entropy.

Notice that the detailed action sampling and evaluation process is explained in the PPO section and will not be repeated here.

== Update
The update step is performed once every 500 steps in practice, key concept is using return values calculated by reward function to update the policy network.

=== Return Value
The return value is calculated and normalized by the following equation:

$
R_t = sum(r_t * gamma^(t-1))
$
$
  R^hat_t = R_t - mu_R / (sigma_R + epsilon)
$

where $R_t$ is the return value of the current state, $r_t$ is the reward of the current state, $gamma$ is the discount factor, which is set to 0.975 in practice.

$mu_R$ is the mean value of the return value, $sigma_R$ is the standard deviation of the return value, $epsilon$ is a small number to prevent division by zero,
which is set to $10^(-8)$ in practice.

=== Policy Loss
The policy loss is associated with the return value, the calculation follows equation:

$
L_p = R^hat_t times log p(a_t)
$
where $L_p$ is the policy loss, $R^hat_t$ is the normalized return value of the current state, $p(a_t)$ is the probability of the action.

=== Total Loss
The total loss is the sum of the policy loss and the entropy loss, the calculation follows equation:

$
L = L_p - e_c times "entropy"(a_t)
$
where $L$ is the total loss, $e_c$ is the entropy coefficient.


Notice that in practice, the entropy coefficient is set to 0.05.


== Training Results
The PGM agent is trained with 120,000 steps since it converged with extremely high speed. The recorded loss curve is shown in the following figure:

#figure(
  image("images/paper/pgm_loss.png", fit: "stretch"),
  caption: [Loss Curve],
) <loss_curve>

The detailed training reward and actual tactical development of the agent will be shown in the result comparison section.

= Discussion

Basically, only Deep-Q-learning agent performs intellectually in the training process, during the training process, the agent has noticed how the shooting system works and has developed the behaviour to actively attack the enemy pawn. Althgouh the agent then converges to stay still at a specific position, but this mis-behaviour is due to the reward setting and can be fixed by adjusting the reward function.

As for PPO and Policy Gradient Method, they failed to form strategies, which is probabiliy due to the unique feature of the game. In the game, only the moving is deterministic, any other non-deterministic states will drastically increase the actual state space the agent percepts, in others words, the hidden states space might actually much larger than we used to expect. This causes the agents needs tons of exploration to find the optimal strategy. Our DQN has a epsilon decay mechanism to directly control the exploration behaviour, while PPO and PGM doesn't have such mechanism, which might be the reason why they failed to form strategies. 

Also, since many state transitions in the game are non-deterministic, the estimation of the return value is a huge challenge, which might significantly affect the training process of PPO and PGM, including DQN. Thus, many hyperparameters could not be tuned in a way that under other environments we could have done. For example, the discount factor is set to 0.97 in PPO and PGM, which is reasonable under many circumstances, but might be relatively high in this game. 


= Attribution

The original code of the paper is open-sourced on github, the link and contributions are as follows:

=== Agents 

Contribution History:

#figure(
  image("images/paper/server-contrib.png", height: 18%, width: 80%, fit: "stretch"),
  caption: [Server Contribution],
) <server_contrib>

=== Paper https://github.com/zivmax/rimworld-combat-agent-paper/graphs/contributors

Contribution History:

#figure(
  image("images/paper/paper-contrib.png", height: 18%, width: 80%,fit: "stretch"),
  caption: [Paper Contribution],
) <paper_contrib>

=== RimWorld Mod https://github.com/zivmax/rimworld-combat-agent-client

Contribution History:

#figure(
  image("images/paper/client-contrib.png", height: 18%, width: 80%,fit: "stretch"),
  caption: [Client Contribution],
) <client_contrib>

The contributors are:

- zivmax: Linzheng Tang 
- mike2367: Guo Yu 

= Acknowledgement

External library usage:

=== PyTorch
[@torch] Neural Networks used in agents are built by both of us from scratch.

=== Gymnasium
[@gymnasium] The training environment is designed and built by both of us based on Gymnasium's vectorized environment from scratch.

=== Viztracer
[@viztracer] This tracing tool helps us to the performance bottleneck of the agents.

=== Pandas
[@pandas] This library is used to store and analyze the training data.

=== Numpy
[@numpy] This library is used to store and process the training data.

=== Matplotlib
[@matplotlib] This library is used to plot the training loss curve.


Notice that due to legal issues, we can't provide the original version of the game which we purchased. If you are interested in recurring the training process, please purchase the genuine game from https://rimworldgame.com/.

// Display bibliography.
#bibliography("refs.bib") 






