# Mathematical Framework for Narrative Dynamics

## 1. Fundamental Representations

### 1.1 Network Structure
Let $G = (V, E, W)$ represent the social network where:
- $V = \{a_1, a_2, ..., a_n\}$ is the set of agents
- $E \subseteq V \times V$ is the set of edges
- $W: E \rightarrow \mathbb{R}^+$ assigns weights to edges

The adjacency matrix $A \in \mathbb{R}^{n \times n}$ represents connections:
$$A_{ij} = \begin{cases}
w_{ij} & \text{if } (a_i, a_j) \in E \\
0 & \text{otherwise}
\end{cases}$$

### 1.2 Narrative Representation
Each narrative $N_i(t)$ for agent $i$ at time $t$ is represented as a vector in embedding space $\mathbb{R}^d$:
$$N_i(t) \in \mathbb{R}^d$$

The ground truth narrative is denoted as $T \in \mathbb{R}^d$.

### 1.3 Resource Distribution
Agent resources (influence capital) are represented as:
$$R_i(t) \in \mathbb{R}^+$$

## 2. Core Dynamics Equations

### 2.1 Narrative Similarity
The similarity between narratives is measured using cosine similarity:
$$\text{sim}(N_i, N_j) = \frac{N_i \cdot N_j}{||N_i|| \cdot ||N_j||}$$

### 2.2 Imperviousness Function
Agents' resistance to influence is modeled using a generalized function:
$$\text{Imp}_i(t) = \frac{(R_i(t))^\alpha}{(R_i(t))^\alpha + K^\alpha} + \delta_i \cdot \text{Tech}_i$$

Where:
- $\alpha$ controls the steepness of the resource-imperviousness relationship
- $K$ is the resource level at which imperviousness = 0.5
- $\delta_i$ represents additional imperviousness due to technical expertise
- $\text{Tech}_i \in \{0,1\}$ indicates technical expertise status

### 2.3 Influence Weight Function
The influence weight of agent $j$ on agent $i$ at time $t$ is:

$$w_{ij}(t) = \sigma\left(\beta_1 \cdot A_{ij} + \beta_2 \cdot \text{sim}(N_i(t), N_j(t)) + \beta_3 \cdot \log\left(\frac{R_j(t)}{R_i(t)}\right) + \beta_4 \cdot \text{Cred}_{ij}(t) - \beta_5 \cdot \text{Imp}_i(t) + \beta_0\right)$$

Where:
- $\sigma$ is the logistic function: $\sigma(x) = \frac{1}{1+e^{-x}}$
- $\text{Cred}_{ij}(t)$ is the perceived credibility of $j$ by $i$
- $\beta_k$ are learned parameters

### 2.4 Narrative Update Equation
The fundamental narrative update equation:

$$N_i(t+1) = N_i(t) + \alpha \sum_{j \in \text{Neigh}(i)} w_{ij}(t) \cdot (N_j(t) - N_i(t)) + \gamma \cdot \text{Ext}_i(t)$$

Where:
- $\alpha$ is the learning rate
- $\text{Neigh}(i)$ is the set of agents connected to $i$
- $\text{Ext}_i(t)$ represents external information inputs
- $\gamma$ is the external information weight

## 3. Hub-and-Spoke Dynamics

### 3.1 Hub Centrality
The hub centrality of an agent $h$ is:
$$\text{HC}(h) = \sum_{i \in V} \sum_{j \in V} \frac{w_{hi}(t) \cdot w_{hj}(t) \cdot \text{sim}(N_i(t), N_j(t))}{1 + \text{sim}(N_i(t), N_j(t))}$$

### 3.2 Spoke Correlation
For agents $i$ and $j$ influenced by hub $h$:
$$\text{SC}(i,j,h) = \frac{\text{cov}(N_i(t+1) - N_i(t), N_j(t+1) - N_j(t))}{w_{hi}(t) \cdot w_{hj}(t) \cdot \sigma_i \cdot \sigma_j}$$

Where $\sigma_i$ and $\sigma_j$ are the standard deviations of narrative changes.

### 3.3 Hub Influence Metric
The overall influence of a hub $h$ on the network:
$$\text{HI}(h) = \sum_{i \in V} w_{hi}(t) \cdot ||N_i(t+1) - N_i(t)||$$

## 4. Multi-System Exploitation Models

### 4.1 Official-Shadow System Divergence
For agent $i$, the divergence between official narrative $N_i^O(t)$ and shadow narrative $N_i^S(t)$:
$$\text{OSD}_i(t) = ||N_i^O(t) - N_i^S(t)||$$

### 4.2 Dual-System Influence
The shadow influence of agent $j$ on agent $i$:
$$w_{ij}^S(t) = w_{ij}(t) \cdot (1 - \text{Pub}_{ij})$$

Where $\text{Pub}_{ij}$ represents the public visibility of the relationship.

### 4.3 Dual-System Effectiveness
The effectiveness of dual-system exploitation:
$$\text{DSE}_i = \frac{||N_i^S(t) - T|| - ||N_i^O(t) - T||}{||T||}$$

## 5. Commitment Trap Dynamics

### 5.1 Commitment Magnitude
The commitment of agent $i$ to narrative $N_i(t)$:
$$\text{CM}_i(t) = \sum_{\tau=0}^{t} \lambda^{t-\tau} \cdot ||N_i(\tau+1) - N_i(\tau)||$$

Where $\lambda \in (0,1)$ is a decay parameter.

### 5.2 Reversal Cost
The cost for agent $i$ to reverse commitment:
$$\text{RC}_i(t) = \text{CM}_i(t) \cdot \text{Vis}_i(t) \cdot \text{Rep}_i(t)$$

Where:
- $\text{Vis}_i(t)$ is the visibility of agent $i$'s commitment
- $\text{Rep}_i(t)$ is the reputation stake

### 5.3 Commitment Trap Probability
The probability that agent $i$ remains trapped in commitment:
$$P(\text{Trapped}_i) = \sigma(\text{RC}_i(t) - \theta_i)$$

Where $\theta_i$ is agent $i$'s threshold for changing position.

## 6. Truth Convergence Metrics

### 6.1 Individual Truth Distance
The distance of agent $i$'s narrative from ground truth:
$$\text{TD}_i(t) = ||N_i(t) - T||$$

### 6.2 System Truth Convergence
The overall system convergence to truth:
$$\text{STC}(t) = \frac{1}{n} \sum_{i=1}^{n} \frac{\text{TD}_i(0) - \text{TD}_i(t)}{\text{TD}_i(0)}$$

### 6.3 Weighted Truth Convergence
Truth convergence weighted by agent influence:
$$\text{WTC}(t) = \frac{\sum_{i=1}^{n} R_i(t) \cdot (\text{TD}_i(0) - \text{TD}_i(t))}{\sum_{i=1}^{n} R_i(t) \cdot \text{TD}_i(0)}$$

## 7. Information Asymmetry Equations

### 7.1 Knowledge Gap
The knowledge gap between agents $i$ and $j$:
$$\text{KG}_{ij}(t) = ||\text{Proj}_T(N_i(t)) - \text{Proj}_T(N_j(t))||$$

Where $\text{Proj}_T$ projects narratives onto dimensions relevant to truth $T$.

### 7.2 Exploitability Index
The exploitability of agent $i$ due to information asymmetry:
$$\text{EI}_i(t) = \sum_{j \in V} \max(0, \text{KG}_{ji}(t) - \text{Imp}_i(t))$$

### 7.3 Strategic Information Advantage
The strategic advantage of agent $i$:
$$\text{SIA}_i(t) = \sum_{j \in V} \max(0, \text{KG}_{ij}(t) - \text{Imp}_j(t))$$

## 8. Resource Dynamics

### 8.1 Resource Update Equation
$$R_i(t+1) = R_i(t) + \delta \sum_{j \in V} w_{ij}(t) \cdot ||N_j(t+1) - N_j(t)|| - \epsilon \sum_{j \in V} w_{ji}(t) \cdot ||N_i(t+1) - N_i(t)||$$

Where:
- $\delta$ represents resource gain from influencing others
- $\epsilon$ represents resource loss from being influenced

### 8.2 Reputation Impact
$$\text{Rep}_i(t+1) = \text{Rep}_i(t) + \eta \sum_{j \in V} w_{ji}(t) \cdot \text{sim}(N_j(t), T) \cdot (1 - \text{Imp}_j(t))$$

Where $\eta$ is the reputation update rate.

## 9. Algorithmic Amplification

### 9.1 Algorithmic Exposure Function
The algorithm-mediated exposure of narrative $j$ to agent $i$:
$$\text{AE}_{ij}(t) = \sigma\left(\gamma_1 \cdot \text{sim}(N_i(t), N_j(t)) + \gamma_2 \cdot \text{Eng}_j(t) + \gamma_3 \cdot \text{Pop}_j(t) - \gamma_4 \cdot \text{Nov}_j(t)\right)$$

Where:
- $\text{Eng}_j(t)$ is the engagement level with narrative $j$
- $\text{Pop}_j(t)$ is the popularity of narrative $j$
- $\text{Nov}_j(t)$ is the novelty of narrative $j$

### 9.2 Filter Bubble Metric
The filter bubble strength for agent $i$:
$$\text{FB}_i(t) = 1 - \frac{\text{Entropy}(\{\text{AE}_{ij}(t)\}_{j \in V})}{\log(|V|)}$$

### 9.3 Algorithmic Reinforcement
The algorithmic reinforcement of narrative $N_i$:
$$\text{AR}_i(t) = \frac{\sum_{j \in V} \text{AE}_{ji}(t) \cdot \text{sim}(N_j(t), N_i(t))}{\sum_{j \in V} \text{AE}_{ji}(t)}$$

## 10. Detecting Coordinated Manipulation

### 10.1 Coordination Score
The coordination score between agents $i$ and $j$:
$$\text{CS}_{ij}(t) = \frac{\text{cov}(N_i(t+1) - N_i(t), N_j(t+1) - N_j(t))}{\sigma_i \cdot \sigma_j}$$

### 10.2 Temporal Coordination
The temporal coordination between agents $i$ and $j$:
$$\text{TC}_{ij}(t) = \exp\left(-\frac{|\tau_i - \tau_j|^2}{2\sigma_t^2}\right)$$

Where:
- $\tau_i$ is the timestamp of agent $i$'s narrative update
- $\sigma_t$ is the temporal bandwidth parameter

### 10.3 Anomalous Coordination Detection
The anomaly score for coordination:
$$\text{ACD}_{ij}(t) = \text{CS}_{ij}(t) \cdot \text{TC}_{ij}(t) \cdot (1 - \text{sim}(N_i(t-1), N_j(t-1)))$$

## 11. Veracity Assessment

### 11.1 Narrative Veracity
The veracity of narrative $N_i$ relative to truth $T$:
$$\text{Ver}_i(t) = \exp\left(-\frac{||N_i(t) - T||^2}{2\sigma_v^2}\right)$$

Where $\sigma_v$ is a scaling parameter.

### 11.2 Source Reliability
The reliability of agent $i$ as an information source:
$$\text{SR}_i(t) = \frac{1}{t} \sum_{\tau=1}^{t} \text{Ver}_i(\tau)$$

### 11.3 Bayesian Veracity Update
The posterior probability of narrative $N_i$ being true:
$$P(N_i(t) | \text{Evidence}) \propto P(\text{Evidence} | N_i(t)) \cdot P(N_i(t))$$

## 12. Network Resilience Metrics

### 12.1 Narrative Diversity
The diversity of narratives in the network:
$$\text{ND}(t) = \frac{1}{n(n-1)} \sum_{i=1}^{n} \sum_{j \neq i}^{n} (1 - \text{sim}(N_i(t), N_j(t)))$$

### 12.2 Echo Chamber Coefficient
The echo chamber coefficient for agent $i$:
$$\text{ECC}_i(t) = \frac{\sum_{j \in \text{Neigh}(i)} w_{ij}(t) \cdot \text{sim}(N_i(t), N_j(t))}{\sum_{j \in \text{Neigh}(i)} w_{ij}(t)}$$

### 12.3 Network Polarization
The overall polarization of the network:
$$\text{Pol}(t) = \frac{\sum_{i=1}^{n} \sum_{j>i}^{n} ||\text{Proj}_1(N_i(t) - N_j(t))||}{\binom{n}{2}}$$

Where $\text{Proj}_1$ projects onto the first principal component of narrative differences.
