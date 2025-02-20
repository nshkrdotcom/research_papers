# Psychologically-Enhanced Narrative Dynamics Framework

Based on your insight about mechanical thinking versus psychological understanding, I'll enhance the mathematical framework with psychological principles that are often overlooked in technical models.

# Psychologically-Enhanced Narrative Dynamics Framework

## 1. Cognitive Biases in Narrative Processing

### 1.1 Confirmation Bias Amplification
The tendency to disproportionately weight information that confirms existing beliefs:

$$N_i(t+1) = N_i(t) + \alpha \sum_{j \in \text{Neigh}(i)} w_{ij}(t) \cdot \text{CB}_i(N_j(t), N_i(t)) \cdot (N_j(t) - N_i(t)) + \gamma \cdot \text{Ext}_i(t)$$

Where the confirmation bias function is:
$$\text{CB}_i(N_j, N_i) = 1 + \beta_{\text{cb}} \cdot \max(0, \text{sim}(N_j, N_i) - \theta_{\text{cb}})$$

### 1.2 Backfire Effect
The tendency for corrections to strengthen originally held misconceptions:

$$\text{Backfire}_i(N_j, N_i) = \begin{cases}
1 - \kappa \cdot \text{sim}(N_j, N_i) & \text{if } \text{sim}(N_j, T) > \text{sim}(N_i, T) \text{ and } \text{OutGroup}(i,j) \\
1 & \text{otherwise}
\end{cases}$$

Where $\kappa$ is the backfire strength parameter and $\text{OutGroup}(i,j)$ indicates whether agent $j$ is perceived as an outgroup member by agent $i$.

### 1.3 Cognitive Dissonance
The discomfort experienced when holding contradictory beliefs:

$$\text{CD}_i(t) = \sum_{j,k \in \text{Beliefs}_i(t)} \max(0, \text{sim}(N_j, N_k) - \theta_{\text{cd}})$$

Where $\text{Beliefs}_i(t)$ is the set of belief components held by agent $i$.

## 2. Emotional Components of Narrative Influence

### 2.1 Affective Influence Weight
Incorporating emotional arousal into influence calculations:

$$w_{ij}^{\text{affect}}(t) = w_{ij}(t) \cdot (1 + \rho \cdot \text{Arousal}(N_j(t)))$$

Where:
- $\text{Arousal}(N)$ measures the emotional intensity of narrative $N$
- $\rho$ is the affective amplification parameter

### 2.2 Emotional Contagion
Modeling how emotions spread through narratives:

$$\text{Emotion}_i(t+1) = (1-\omega_e) \cdot \text{Emotion}_i(t) + \omega_e \sum_{j \in \text{Neigh}(i)} \frac{w_{ij}(t) \cdot \text{Emotion}_j(t)}{\sum_{k \in \text{Neigh}(i)} w_{ik}(t)}$$

Where $\omega_e$ is the emotional update weight.

### 2.3 Moral Foundations Alignment
Influence based on moral foundation alignment:

$$\text{MFA}_{ij}(t) = \exp\left(-\frac{||\text{MF}_i - \text{MF}_j||^2}{2\sigma_{\text{mf}}^2}\right)$$

Where:
- $\text{MF}_i$ is agent $i$'s moral foundations vector
- $\sigma_{\text{mf}}$ is a scaling parameter

## 3. Social Identity and Group Dynamics

### 3.1 Social Identity Salience
The prominence of group identity in narrative processing:

$$\text{SIS}_i(t) = \frac{\sum_{g \in \text{Groups}_i} \text{Threat}_g(t) \cdot \text{Centrality}_{ig}}{\sum_{g \in \text{Groups}_i} \text{Centrality}_{ig}}$$

Where:
- $\text{Groups}_i$ is the set of groups agent $i$ identifies with
- $\text{Threat}_g(t)$ is the perceived threat to group $g$
- $\text{Centrality}_{ig}$ is the centrality of group $g$ to agent $i$'s identity

### 3.2 Ingroup-Outgroup Influence Modulation
Different influence dynamics based on group membership:

$$w_{ij}^{\text{group}}(t) = \begin{cases}
w_{ij}(t) \cdot (1 + \mu_{\text{in}}) & \text{if } \text{InGroup}(i,j) \\
w_{ij}(t) \cdot (1 - \mu_{\text{out}} \cdot \text{SIS}_i(t)) & \text{if } \text{OutGroup}(i,j)
\end{cases}$$

Where:
- $\mu_{\text{in}}$ is the ingroup bonus parameter
- $\mu_{\text{out}}$ is the outgroup penalty parameter

### 3.3 Social Conformity Pressure
The pressure to conform to group norms:

$$\text{SCP}_i(g, t) = \phi \cdot \frac{|\{j \in \text{Group}_g : \text{sim}(N_j(t), \bar{N}_g(t)) > \theta_{\text{conf}}\}|}{|\text{Group}_g|}$$

Where:
- $\bar{N}_g(t)$ is the average narrative of group $g$
- $\phi$ is the conformity pressure parameter
- $\theta_{\text{conf}}$ is the conformity threshold

## 4. Trust and Credibility Dynamics

### 4.1 Trust Accumulation Model
How trust builds and decays over time:

$$\text{Trust}_{ij}(t+1) = (1-\delta_T) \cdot \text{Trust}_{ij}(t) + \delta_T \cdot \text{TrustUpdate}_{ij}(t)$$

Where:
$$\text{TrustUpdate}_{ij}(t) = \begin{cases}
\tau_{\text{pos}} \cdot (1 - \text{Trust}_{ij}(t)) & \text{if } \text{Credible}_j(t) \\
-\tau_{\text{neg}} \cdot \text{Trust}_{ij}(t) & \text{if } \neg\text{Credible}_j(t)
\end{cases}$$

And:
- $\delta_T$ is the trust update rate
- $\tau_{\text{pos}}$ is the positive trust reinforcement rate
- $\tau_{\text{neg}}$ is the negative trust erosion rate
- $\text{Credible}_j(t)$ indicates whether agent $j$'s narrative at time $t$ is credible

### 4.2 Source Credibility Heuristics
How credibility judgments incorporate cognitive shortcuts:

$$\text{Cred}_{ij}(t) = \alpha_{\text{exp}} \cdot \text{Expertise}_j + \alpha_{\text{trust}} \cdot \text{Trust}_{ij}(t) + \alpha_{\text{sim}} \cdot \text{sim}(N_i(t), N_j(t)) + \alpha_{\text{group}} \cdot \text{InGroup}(i,j)$$

Where $\alpha$ parameters weight different credibility components.

### 4.3 Epistemic Authority
The perceived right to make knowledge claims:

$$\text{EA}_{ij}(t) = \text{Cred}_{ij}(t) \cdot (1 + \epsilon_{\text{status}} \cdot \text{StatusDiff}_{ij} + \epsilon_{\text{cert}} \cdot \text{Certainty}_j(t))$$

Where:
- $\text{StatusDiff}_{ij}$ is the status difference between agents
- $\text{Certainty}_j(t)$ is agent $j$'s expressed certainty
- $\epsilon$ parameters control the impact of status and certainty

## 5. Cognitive Load and Processing Depth

### 5.1 Cognitive Resource Allocation
How limited cognitive resources affect narrative processing:

$$\text{CogLoad}_i(t) = \min\left(1, \frac{\sum_{j \in \text{Inputs}_i(t)} \text{Complexity}(N_j(t))}{\text{CogCapacity}_i}\right)$$

Where:
- $\text{Inputs}_i(t)$ is the set of narratives agent $i$ is exposed to
- $\text{Complexity}(N)$ measures narrative complexity
- $\text{CogCapacity}_i$ is agent $i$'s cognitive processing capacity

### 5.2 Elaboration Likelihood
The probability of deep vs. peripheral processing:

$$P(\text{Deep}_i(t)) = (1 - \text{CogLoad}_i(t)) \cdot \text{Motivation}_i(t)$$

Where $\text{Motivation}_i(t)$ represents the motivation to process deeply.

### 5.3 Heuristic Processing Influence
Influence under heuristic processing conditions:

$$w_{ij}^{\text{heur}}(t) = \begin{cases}
w_{ij}(t) & \text{if } P(\text{Deep}_i(t)) > \theta_{\text{elab}} \\
\text{EA}_{ij}(t) \cdot (1 + \psi \cdot \text{Affect}_j(t)) & \text{otherwise}
\end{cases}$$

Where:
- $\theta_{\text{elab}}$ is the elaboration threshold
- $\psi$ is the affective heuristic parameter
- $\text{Affect}_j(t)$ is the affective content of narrative $N_j(t)$

## 6. Psychological Reactance and Boomerang Effects

### 6.1 Perceived Freedom Threat
The degree to which influence attempts are perceived as threatening autonomy:

$$\text{PFT}_{ij}(t) = \xi_1 \cdot \text{Intensity}_{ij}(t) + \xi_2 \cdot \text{Explicit}_{ij}(t) + \xi_3 \cdot \text{OutGroup}(i,j)$$

Where:
- $\text{Intensity}_{ij}(t)$ is the perceived intensity of influence attempt
- $\text{Explicit}_{ij}(t)$ is the explicitness of influence attempt
- $\xi$ parameters weight different threat components

### 6.2 Reactance Response
How reactance affects influence:

$$w_{ij}^{\text{react}}(t) = w_{ij}(t) \cdot (1 - \chi \cdot \text{PFT}_{ij}(t) \cdot \text{ReactanceProne}_i)$$

Where:
- $\chi$ is the reactance strength parameter
- $\text{ReactanceProne}_i$ is agent $i$'s proneness to reactance

### 6.3 Boomerang Effect
Movement in the opposite direction of influence attempt:

$$\text{Boomerang}_{ij}(t) = \begin{cases}
-\zeta \cdot ||N_j(t) - N_i(t)|| & \text{if } \text{PFT}_{ij}(t) > \theta_{\text{react}} \\
0 & \text{otherwise}
\end{cases}$$

Where:
- $\zeta$ is the boomerang strength parameter
- $\theta_{\text{react}}$ is the reactance threshold

## 7. Self-Perception and Narrative Identity

### 7.1 Narrative Self-Consistency
The pressure to maintain a coherent self-narrative:

$$\text{NSC}_i(t) = \exp\left(-\frac{||N_i(t) - N_i(t-1)||^2}{2\sigma_{\text{self}}^2 \cdot \text{Identity}_i(N_i(t))}\right)$$

Where:
- $\sigma_{\text{self}}$ is a scaling parameter
- $\text{Identity}_i(N)$ measures how central narrative $N$ is to agent $i$'s identity

### 7.2 Self-Perception Influence
How behavior affects self-perception:

$$\text{SP}_i(t) = \sum_{\tau=t-k}^{t-1} \lambda^{t-\tau} \cdot \text{Behavior}_i(\tau)$$

Where:
- $\text{Behavior}_i(t)$ represents agent $i$'s behavior at time $t$
- $\lambda$ is a decay parameter
- $k$ is the history window

### 7.3 Identity-Protective Cognition
Resistance to identity-threatening information:

$$w_{ij}^{\text{identity}}(t) = w_{ij}(t) \cdot \exp\left(-\nu \cdot \text{IdentityThreat}_i(N_j(t))\right)$$

Where:
- $\nu$ is the identity protection parameter
- $\text{IdentityThreat}_i(N)$ measures how threatening narrative $N$ is to agent $i$'s identity

## 8. Dual-Process Narrative Updating

### 8.1 System 1 (Fast) Processing
Automatic, intuitive narrative updates:

$$N_i^{S1}(t+1) = N_i(t) + \alpha_{S1} \sum_{j \in \text{Neigh}(i)} w_{ij}^{\text{affect}}(t) \cdot w_{ij}^{\text{group}}(t) \cdot (N_j(t) - N_i(t))$$

### 8.2 System 2 (Slow) Processing
Deliberative, analytical narrative updates:

$$N_i^{S2}(t+1) = N_i(t) + \alpha_{S2} \sum_{j \in \text{Neigh}(i)} w_{ij}(t) \cdot \text{EvidenceWeight}(N_j(t)) \cdot (N_j(t) - N_i(t))$$

Where $\text{EvidenceWeight}(N)$ evaluates the evidential support for narrative $N$.

### 8.3 Integrated Dual-Process Update
The combined narrative update:

$$N_i(t+1) = (1 - P(\text{Deep}_i(t))) \cdot N_i^{S1}(t+1) + P(\text{Deep}_i(t)) \cdot N_i^{S2}(t+1)$$

## 9. Psychological Defense Mechanisms

### 9.1 Denial Coefficient
The tendency to reject threatening information:

$$\text{Denial}_i(N_j) = \min\left(1, \frac{\text{Threat}_i(N_j)}{\theta_{\text{threat}} + \text{Resilience}_i(t)}\right)$$

Where:
- $\text{Threat}_i(N_j)$ is the perceived threat of narrative $N_j$ to agent $i$
- $\theta_{\text{threat}}$ is the threat threshold
- $\text{Resilience}_i(t)$ is agent $i$'s psychological resilience

### 9.2 Rationalization Function
How contradictory information is reconciled:

$$\text{Rational}_i(N_j, N_i) = \frac{1}{1 + \exp(-\sigma_r \cdot (\text{sim}(N_j, N_i) - \theta_r))}$$

Where:
- $\sigma_r$ controls the steepness of the rationalization function
- $\theta_r$ is the rationalization threshold

### 9.3 Projection Mechanism
Attribution of one's characteristics to others:

$$\text{Proj}_{ij}(t) = \pi \cdot \text{sim}(N_i(t-\Delta t), N_j(t)) \cdot \text{NegAffect}_i(t)$$

Where:
- $\pi$ is the projection parameter
- $\Delta t$ is the time lag
- $\text{NegAffect}_i(t)$ is agent $i$'s negative affect state

## 10. Narrative Framing Effects

### 10.1 Loss-Gain Asymmetry
Different responses to loss-framed versus gain-framed narratives:

$$\text{FrameWeight}_i(N_j) = \begin{cases}
1 + \lambda_{\text{loss}} & \text{if } \text{LossFrame}(N_j) \\
1 + \lambda_{\text{gain}} & \text{if } \text{GainFrame}(N_j)
\end{cases}$$

Where $\lambda_{\text{loss}} > \lambda_{\text{gain}}$ reflects loss aversion.

### 10.2 Narrative Transportation
Immersion in narrative reducing critical evaluation:

$$\text{Transport}_i(N_j) = \frac{1}{1 + \exp(-\sigma_t \cdot (\text{Vividness}(N_j) + \text{Coherence}(N_j) - \theta_t))}$$

Where:
- $\sigma_t$ controls the steepness of the transportation function
- $\theta_t$ is the transportation threshold
- $\text{Vividness}(N_j)$ and $\text{Coherence}(N_j)$ measure narrative qualities

### 10.3 Metaphorical Framing Influence
How metaphors shape narrative understanding:

$$\text{MetaInfluence}_i(N_j) = \omega_m \cdot \text{MetaphorActivation}_i(N_j) \cdot \text{ConceptualMapping}(N_j)$$

Where:
- $\omega_m$ is the metaphorical influence weight
- $\text{MetaphorActivation}_i(N_j)$ is the activation strength of the metaphor for agent $i$
- $\text{ConceptualMapping}(N_j)$ measures how strongly the metaphor maps to the target domain

The mathematical framework I've developed integrates core psychological principles that mechanical "first principles" approaches often miss. This enhancement addresses several critical psychological dimensions:

## 1. Cognitive Biases and Heuristics

The model now incorporates:
- **Confirmation bias** with a mathematical function that amplifies information aligning with existing beliefs
- **Backfire effect** that increases resistance when corrections come from outgroup sources
- **Cognitive dissonance** quantified as tension between contradictory beliefs

For example, the confirmation bias function:
```
CB_i(N_j, N_i) = 1 + β_cb · max(0, sim(N_j, N_i) - θ_cb)
```
Shows how information consistent with existing beliefs receives disproportionate weight.

## 2. Emotional and Moral Dimensions

The enhanced framework acknowledges that humans aren't purely rational:
- **Affective influence** captures how emotionally arousing content receives more attention
- **Emotional contagion** models how emotions spread through networks
- **Moral foundations alignment** quantifies how shared values increase influence

This addresses a critical blind spot in mechanical models - that moral and emotional resonance often outweighs logical coherence in human reasoning.

## 3. Social Identity and Group Dynamics

The model now accounts for:
- **Social identity salience** that increases during perceived threat
- **Ingroup-outgroup processing** with different influence weights based on group membership
- **Conformity pressure** that increases with group consensus

This captures how people process identical information differently based on social context - a dynamic purely mechanical models miss entirely.

## 4. Psychological Reactance and Defense Mechanisms

The framework incorporates resistance mechanisms:
- **Psychological reactance** when influence attempts threaten perceived freedom
- **Identity-protective cognition** that filters threatening information
- **Denial and rationalization** functions that maintain existing worldviews

These equations model why direct confrontation often backfires, while indirect influence may succeed - crucial psychological dynamics that mechanical approaches overlook.

## 5. Dual-Process Integration

By modeling both System 1 (fast, intuitive) and System 2 (slow, deliberative) processing, we capture:
- How cognitive load reduces deliberative processing
- Different pathways for emotional versus analytical influence
- Conditions under which heuristic shortcuts dominate reasoning

This improves on mechanical models that assume uniform information processing regardless of context.

---

These psychologically-informed enhancements create a framework that better reflects how real humans process information, form beliefs, and resist change. The equations provide testable predictions about narrative dynamics that account for both the rational and non-rational aspects of human psychology - addressing precisely the limitations of mechanical "first principles" thinking.
