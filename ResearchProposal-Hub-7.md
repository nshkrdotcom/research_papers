I appreciate the detailed function definitions you've provided. Let me integrate these with my previous psychological framework and create a unified mathematical model that's both psychologically grounded and computationally implementable.

# Unified Psychological Framework for Narrative Dynamics

This framework integrates psychological principles into mathematical representations of narrative influence and evolution, addressing the limitations of purely mechanical "first principles" approaches.

## 1. Cognitive Bias Integration

### 1.1 Confirmation Bias
```
ConfirmationBias(N_i, N_j) = max(0, sim(N_i, N_j) - θ_cb) * PriorBeliefStrength(i, t)
```
Where:
- `sim(N_i, N_j)` is the cosine similarity between narratives
- `θ_cb` is the confirmation threshold
- `PriorBeliefStrength(i, t)` measures how strongly agent i holds their beliefs

### 1.2 Simplified Confirmation Bias (Implementation-Ready)
```
ConfirmationBias(N_i, N_j) = {
    bias_i,                 if sim(N_i, N_j) ≥ 0.7
    -bias_j,                if sim(N_i, N_j) < 0.4
    0,                      otherwise
}
```

### 1.3 Disconfirmation Discount
```
DisconfirmationDiscount(N_i, N_j, PBS) = 
    sim(N_i, N_j) < 0 ? 
    PBS * (1 - |sim(N_i, N_j)|) : 
    0
```
Where PBS (Prior Belief Strength) is calculated as:
```
PriorBeliefStrength(i, t) = 1 - (AverageDistanceToTruth(i, t_0:t) / MaxPossibleDistance)
```

### 1.4 Cognitive Dissonance
```
CognitiveDissonance(i, t) = ∑_{a,b ∈ Beliefs_i} |sim(N_a, N_b) - 0.5|^2
```
Where higher values indicate greater internal contradictions in belief system.

## 2. Emotional Processing

### 2.1 Emotion Detection
```
E_i(t) = EmotionModel(N_i(t))
```
Where `EmotionModel()` is a trained classifier that returns an emotion vector:
```
E_i(t) = [fear_i(t), anger_i(t), joy_i(t), trust_i(t), disgust_i(t)]
```

### 2.2 Emotional Congruence
```
EmotionalCongruence(E_i, E_j) = cos(E_i, E_j)
```

### 2.3 Emotional Intensity Amplification
```
EmotionalIntensity(E) = ||E||_2 / √n
```
Where n is the dimensionality of the emotion vector.

### 2.4 Affect-Based Processing Mode
```
P(AffectiveProcessing_i) = min(1, σ(EmotionalIntensity(E_i) - θ_affect))
```

## 3. Social Identity and Group Dynamics

### 3.1 Group Identity
```
GroupIdentity(i, j) = {
    1,    if Group(i) = Group(j)
    0,    otherwise
}
```

### 3.2 Practical Social Media Implementation
```
GroupIdentity(i, j) = {
    B,    if j ∈ user_i_friend_list
    0,    otherwise
}
```
Where B is a bias constant.

### 3.3 Threat-Based Identity Salience
```
IdentitySalience(i, t) = BaseIdentity_i + ThreatAmplification(i, t)
```

### 3.4 Group Conformity Pressure
```
ConformityPressure(i, g, t) = ϕ * (|{j ∈ Group_g : sim(N_j, Ñ_g) > θ_conf}| / |Group_g|)
```
Where:
- `Ñ_g` is the average narrative of group g
- `ϕ` is the conformity coefficient
- `θ_conf` is the conformity threshold

## 4. Authority and Credibility

### 4.1 Authority Measurement
```
Authority(j) = Normalize(PageRank(j))
```
Where `Normalize()` scales values to [0,1] range across the network.

### 4.2 Perceived Credibility
```
PerceivedCredibility(j, t) = α_exp * Expertise(j) + 
                            α_rep * Reputation(j, t) + 
                            α_auth * Authority(j) + 
                            α_group * InGroup(i, j)
```

### 4.3 Simple Credibility Approximation
```
PerceivedCredibility(j, t) = ||N_j(t) - T|| / ||T||
```
Where T represents ground truth.

## 5. Trust Dynamics

### 5.1 Trust Update Equation
```
Trust(i, j, t+1) = Trust(i, j, t) + κ * (Veracity(N_j(t)) - λ * (1-Veracity(N_j(t))))
```
Where:
- `κ` is the trust learning rate
- `λ` is the penalty for untrustworthy narratives
- `Veracity()` measures alignment with truth

### 5.2 Relationship History Integration
```
Trust(i, j, t) = (1-δ_t) * BaselineTrust(i, j) + 
                δ_t * (∑_{τ=1}^{t-1} γ^{t-τ} * InteractionQuality(i, j, τ))
```
Where:
- `δ_t` weights history vs baseline
- `γ` is the decay parameter for historical interactions
- `InteractionQuality()` measures the value of past interactions

## 6. Cognitive Resources and Attention

### 6.1 Cognitive Load
```
CognitiveLoad(i, t) = min(1, ∑_{j ∈ Inputs_i(t)} Complexity(N_j(t)) / CogCapacity_i)
```

### 6.2 Processing Depth Probability
```
P(DeepProcessing_i) = (1 - CognitiveLoad(i, t)) * Motivation(i, t) * (1 - Fatigue(i, t))
```

### 6.3 Attention Allocation
```
Attention(i, j, t) = Salience(N_j, t) * (1 - CognitiveLoad(i, t)) * Interest(i, N_j)
```
Where:
- `Salience()` captures how attention-grabbing content is
- `Interest()` models agent i's topical interests

## 7. Narrative Coherence

### 7.1 Coherence Function
```
Coherence(N) = α_causal * CausalConsistency(N) + 
              α_struct * StructuralCompleteness(N) + 
              α_fact * FactualAccuracy(N)
```

### 7.2 Coherence Delta
```
NarrativeCoherenceDelta(i, j, t) = Coherence(N_i(t) + ΔN_i(j, t)) - Coherence(N_i(t))
```
Where `ΔN_i(j, t)` represents the influence from agent j.

### 7.3 Simplified Coherence Impact
```
CoherenceImpact(i, j, t) = {
    positive_bias,    if NarrativeCoherenceDelta(i, j, t) > 0
    negative_bias,    otherwise
}
```

## 8. Psychological Reactance

### 8.1 Freedom Threat Perception
```
FreedomThreat(i, j, t) = ξ_1 * InfluenceIntensity(j, t) + 
                         ξ_2 * Explicitness(N_j, t) + 
                         ξ_3 * OutGroup(i, j)
```

### 8.2 Reactance Response
```
ReactanceEffect(i, j, t) = -ψ * FreedomThreat(i, j, t) * ReactanceProne(i)
```
Where `ReactanceProne(i)` is agent i's tendency toward reactance.

### 8.3 Boomerang Effect
```
BoomerangEffect(i, j, t) = {
    -ζ * ||N_j(t) - N_i(t)||,    if FreedomThreat(i, j, t) > θ_react
    0,                          otherwise
}
```

## 9. Defensive Processing

### 9.1 Identity Threat Detection
```
IdentityThreat(i, N_j) = ||Proj_identity(N_j) - Proj_identity(N_i)|| * IdentityCentrality(i)
```
Where:
- `Proj_identity()` projects narrative onto identity-relevant dimensions
- `IdentityCentrality()` measures how central beliefs are to identity

### 9.2 Denial Probability
```
P(Denial_i(N_j)) = σ(IdentityThreat(i, N_j) - DefensiveThreshold(i))
```

### 9.3 Rationalization Effect
```
RationalizationEffect(i, j, t) = min(1, IdentityThreat(i, N_j) / θ_rational) * SimilarityBias
```
Where `SimilarityBias` increases perceived similarity with preferred narratives.

## 10. Integrated Influence Weight Function

The complete influence weight function incorporating all psychological components:

```
w(i, j, t) = σ(
    β_1 * A[i, j] +
    β_2 * Similarity(N_i(t), N_j(t)) +
    β_3 * ResourceDifference(i, j, t) +
    β_4 * PerceivedCredibility(j, t) +
    β_5 * (1-Imp_i(t)) +
    β_6 * ConfirmationBias(N_i(t), N_j(t)) +
    β_7 * EmotionalCongruence(E_i(t), E_j(t)) +
    β_8 * GroupIdentity(i, j) * IdentitySalience(i, t) +
    β_9 * Authority(j) * (1 - ReactanceEffect(i, j, t)) +
    β_10 * (1 - DisconfirmationDiscount(N_i(t), N_j(t), PriorBeliefStrength(i, t))) +
    β_11 * Trust(i, j, t) +
    β_12 * NarrativeCoherenceDelta(i, j, t) +
    β_13 * (1 - CognitiveLoad(i, t)) +
    β_14 * (1 - P(Denial_i(N_j))) +
    β_15 * BoomerangEffect(i, j, t) +
    c
)
```

## 11. Dual-Process Narrative Update

The complete dual-process narrative update incorporating both rational and emotional pathways:

```
N_i(t+1) = N_i(t) + 
           P(DeepProcessing_i) * α_central * ∑_j w_central(i, j, t) * (N_j(t) - N_i(t)) +
           (1-P(DeepProcessing_i)) * α_peripheral * ∑_j w_peripheral(i, j, t) * (N_j(t) - N_i(t))
```

Where:
- `w_central()` weights evidence quality and logical consistency
- `w_peripheral()` weights emotional appeal and source characteristics
- `α_central` and `α_peripheral` are learning rates for each pathway

## 12. Implementation Considerations

For computational efficiency and initial testing, simplified versions of these equations can be used:

1. Begin with core psychological biases (confirmation bias, emotional congruence, group identity)
2. Use threshold-based approximations instead of continuous functions
3. Focus on measuring a few key psychological variables through proxy metrics
4. Gradually incorporate more complex dynamics as validation progresses

This framework balances psychological realism with computational feasibility, creating a model that captures human factors while remaining mathematically tractable.







# Unified Psychological-Mathematical Framework for Narrative Dynamics

I've developed a comprehensive framework that integrates your detailed function specifications with my psychological enhancements, creating a system that's both mathematically rigorous and psychologically realistic.

## Core Innovations in This Unified Framework

### 1. Tiered Implementation Approach
The framework provides both theoretically complete equations and simplified implementations for practical testing:

```
// Theoretical confirmation bias
ConfirmationBias(N_i, N_j) = max(0, sim(N_i, N_j) - θ_cb) * PriorBeliefStrength(i, t)

// Simplified implementation
ConfirmationBias(N_i, N_j) = {
    bias_i,          if sim(N_i, N_j) ≥ 0.7
    -bias_j,         if sim(N_i, N_j) < 0.4
    0,               otherwise
}
```

This dual representation allows researchers to start with computationally efficient approximations while maintaining theoretical integrity.

### 2. Dual-Process Architecture
The model explicitly distinguishes between fast, automatic (System 1) and slow, deliberative (System 2) processing:

```
N_i(t+1) = N_i(t) + 
           P(DeepProcessing_i) * α_central * ∑_j w_central(i,j,t) * (N_j(t) - N_i(t)) +
           (1-P(DeepProcessing_i)) * α_peripheral * ∑_j w_peripheral(i,j,t) * (N_j(t) - N_i(t))
```

This captures how cognitive load, motivation, and emotional arousal determine whether people process information analytically or heuristically.

### 3. Defensive Processing Mechanisms
The framework includes mathematical representations of psychological defense mechanisms:
- Identity threat detection equations
- Denial probability functions 
- Rationalization effects

These capture how people protect cherished beliefs, explaining why factual corrections often fail.

### 4. Emotion as a Separate Processing Channel
Emotions aren't just "noise" in the system but a parallel processing channel:

```
E_i(t) = [fear_i(t), anger_i(t), joy_i(t), trust_i(t), disgust_i(t)]
EmotionalCongruence(E_i, E_j) = cos(E_i, E_j)
```

This multidimensional emotion representation affects both influence weights and processing depth.

### 5. Dynamic Trust Evolution
Trust is modeled as relationship capital that evolves through interactions:

```
Trust(i,j,t+1) = Trust(i,j,t) + κ * (Veracity(N_j(t)) - λ * (1-Veracity(N_j(t))))
```

This captures how trust is slowly built but can be rapidly destroyed, creating asymmetric influence dynamics.

## Key Advantages Over Previous Approaches

1. **Computational Tractability**: Despite psychological complexity, the equations remain mathematically well-defined and computable

2. **Testable Predictions**: Each component makes specific, testable predictions about narrative dynamics

3. **Modular Design**: Researchers can activate or deactivate specific psychological components to isolate effects

4. **Integration of Social and Cognitive Factors**: Unlike purely cognitive or social models, this framework integrates both dimensions

5. **Implementation Flexibility**: The tiered approach allows implementation at different levels of complexity depending on computational resources

## Research Applications

This framework enables testing questions like:

1. How do emotional vs. rational appeals differ in their effectiveness for different audience segments?

2. Under what conditions do fact-checking efforts backfire versus succeed?

3. How do network structures interact with psychological biases to create echo chambers?

4. What interventions might increase receptivity to corrective information?

Rather than treating "first principles" and "psychological principles" as separate domains, this framework demonstrates how they can be integrated into a unified mathematical system that maintains computational rigor while capturing the messy realities of human psychology.
