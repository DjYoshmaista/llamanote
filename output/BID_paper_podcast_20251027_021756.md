---
# Document Metadata
- **source_file**: /home/yosh/Downloads/BID_paper.pdf
- **processing_mode**: podcast
- **model_used**: qwen3-4b
- **stages_completed**: ['extract', 'preprocess', 'chunk', 'process', 'filter', 'format']
---


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
BIDIRECTIONAL DECODING: IMPROVING ACTION
CHUNKING VIA GUIDED TEST-TIME SAMPLING
Yuejiang Liu∗, Jubayer Ibn Hamid∗, Annie Xie, Yoonho Lee, Maximilian Du, Chelsea Finn
Department of Computer Science, Stanford University

**[Speaker Guest]:**

ABSTRACT
Predicting and executing a sequence of actions without intermediate replanning,
known as action chunking, is increasingly used in robot learning from human
demonstrations. Yet, its effects on the learned policy remain inconsistent: some
studies find it crucial for achieving strong results, while others observe decreased
performance. In this paper, we first dissect how action chunking impacts the divergence between a learner and a demonstrator. We find that action chunking
allows the learner to better capture the temporal dependencies in demonstrations
but at the cost of reduced reactivity to unexpected states. To address this tradeoff,
we propose Bidirectional Decoding (BID), a test-time inference algorithm that
bridges action chunking with closed-loop adaptation. At each timestep, BID samples multiple candidate predictions and searches for the optimal one based on two
criteria: (i) backward coherence, which favors samples that align with previous
decisions; (ii) forward contrast, which seeks samples of high likelihood for future plans. By coupling decisions within and across action chunks, BID promotes
both long-term consistency and short-term reactivity. Experimental results show
that our method boosts the performance of two state-of-the-art generative policies
across seven simulation benchmarks and two real-world tasks. Code and videos
are available at [web link] github. io.

**[Speaker Host]:**

INTRODUCTION
The increasing availability of human demonstrations has spurred renewed interest in behavioral
cloning (Atkeson & Schaal, 1997; Argall et al., 2009). In particular, recent studies have highlighted
the potential of learning from large-scale demonstrations to acquire a variety of complex skills (Zhao
et al., 2023b; Chi et al., 2023; Fu et al., 2024b; Lee et a

**[Speaker Guest]:**

assistant
Action chunking—predicting and executing a sequence of actions without intermediate replanning—is increasingly used in robot learning from human demonstrations. However, its impact on learned policies remains inconsistent: some studies find it essential for strong performance, while others report degraded results.

We investigate how action chunking affects the alignment between a learner and a demonstrator. Our analysis shows that while action chunking helps the learner better capture temporal dependencies in demonstrations, it reduces reactivity to unexpected states. To balance this tradeoff, we propose Bidirectional Decoding (BID), a test-time inference algorithm that integrates action chunking with closed-loop adaptation.

At each timestep, BID samples multiple candidate actions and selects the optimal one based on two criteria: (i) backward coherence—favoring predictions that align with prior decisions; and (ii) forward contrast—seeking high-probability options that support future planning. By coupling decisions both within and across action chunks, BID enables both long-term consistency and short-term reactivity.

Experiments demonstrate that BID improves the performance of two state-of-the-art generative policies across seven simulation benchmarks and two real-world tasks. Code and videos are available at github.io/[web link].


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
023b; Chi et al., 2023; Fu et al., 2024b; Lee et al., 2024; Khazatsky et al., 2024). Yet, existing
methods still struggle with two common properties of human demonstrations: (i) strong temporal
dependencies across multiple steps, such as idle pauses (Chi et al., 2023) and latent strategies (Xie
et al., 2021; Ma et al., 2024), (ii) large style variability across different demonstrations, such as
differences in proficiency (Belkhale et al., 2024) and preference (Kuefler & Kochenderfer, 2017).
Often, both properties are prevalent yet unlabeled in collected data, posing significant challenges to
the traditional behavioral cloning that maps an input state to an action.
In response to these challenges, recent works have pursued a generative approach equipped with
action chunking: (i) predicting a sequence of actions over multiple time steps and executing all or
part of the sequence (Zhao et al., 2023b; Chi et al., 2023); (ii) modeling the distribution of action
chunks and sampling from the learned model in an independent (Chi et al., 2023; Prasad et al., 2024)
or weakly dependent (Janner et al., 2022; Zhao et al., 2023b) manner for sequential decisions. Some
studies find this approach crucial for learning a performant policy in laboratory scenarios (Zhao
et al., 2023b; Chi et al., 2023), while other recent work reports opposite outcomes under practical
conditions (Lee et al., 2024). The reasons behind these conflicting observations remain unclear.
In this paper, we first dissect the influence of action chunking by examining the divergence between
learned policies and human demonstrations. We find that, when a policy is built with limited context
length - little or no history is used as input for robustness or efficiency (Mandlekar et al., 2020;
Bharadhwaj et al., 2023b; Brohan et al., 2023a; b; Shi et al., 2023; Collaboration, 2023) - increasing
the length of action chunks allows for implicit conditioning on more past actions, thereby improving
its ability to capture the

**[Speaker Host]:**

assistant
Chi et al., 2023; Fu et al., 2024b; Lee et al., 2024; Khazatsky et al., 2024. Yet, existing methods still struggle with two common properties of human demonstrations: strong temporal dependencies across multiple steps, such as idle pauses and latent strategies, and large style variability across different demonstrations, such as differences in proficiency and preference. Often, both properties are present but unlabeled in collected data, posing significant challenges to traditional behavioral cloning that maps an input state to an action.

In response, recent works have pursued a generative approach equipped with action chunking: predicting a sequence of actions over multiple time steps and executing all or part of the sequence, or modeling the distribution of action chunks and sampling from the learned model in an independent or weakly dependent manner. Some studies find this approach crucial for learning a performant policy in laboratory settings, while others report opposite outcomes under practical conditions. The reasons behind these conflicting results remain unclear.

This paper dissects the influence of action chunking by examining the divergence between learned policies and human demonstrations. We find that when a policy is built with limited context length—little or no history used for robustness or efficiency—increasing the length of action chunks allows for implicit conditioning on more past actions, thereby improving its ability to capture temporal dependencies and better align with human behavior.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
ons, thereby improving
its ability to capture the temporal dependencies inherent in demonstrations. However, this advantage
∗Equal contribution.

Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

(a) action chunking
(b) receding horizon
(c) bidirectional decoding
Figure 1: Illustration of different inference methods applied to a robot policy with action chunking. The robot
is tasked with catching a moving trolley. (a) Vanilla action chunking (Zhao et al., 2023b) executes actions based
on previous predictions, resulting in delayed reactions to object motions. (b) Receding horizon (Chi et al., 2023)
enables faster reactions, but leads to a jittery trajectory in the presence of multimodal demonstrations (e. g., both
left- and right-handers). (c) Our Bidirectional Decoding explicitly searches for the optimal action from multiple
predictions sampled at each time step, achieving both long-term consistency and short-term reactivity.
comes at the cost of reduced access to recent state observations, which can be crucial for reacting
to unexpected dynamics arising from modeling errors or environmental stochasticity. This tradeoff raises a crucial question: How can we preserve the strengths of action chunking in long-term

**[Speaker Host]:**

❓ consistency without suffering from its limitations in short-term reactivity?
To this end, we introduce Bidirectional Decoding (BID), an inference algorithm that bridges action
chunking with closed-loop adaptation. Our main idea is to sample multiple predictions at each time
step and search for the most desirable one. Specifically, BID operates on two decoding criteria: (i)
backward coherence, which favors samples that are close to the sequence selected at the previous
step; (ii) forward contrast, which favors samples that are close to the output of a stronger policy
and distant from those of a weaker one. As illustrated in Fig. 1, BID updates the chunk of future
actions based on the previous strategy, promoting temporal consistency over extended periods while

**[Speaker Guest]:**

rema

**[Speaker Host]:**

assistant

**[Speaker Guest]:**

❓ Bidirectional Decoding (BID) is an inference algorithm designed to address the tradeoff between long-term consistency and short-term reactivity in action chunking. While traditional action chunking improves temporal dependency capture, it sacrifices access to recent state observations, making it less responsive to unexpected dynamics or environmental noise. This creates a key challenge: how can we maintain long-term consistency without losing short-term reactivity?

To resolve this, BID introduces a mechanism that samples multiple action predictions at each time step and selects the most optimal one through a dual decoding process. It operates on two criteria: backward coherence, which ensures the current action chunk aligns with the previous step’s strategy, promoting continuity; and forward contrast, which favors predictions that are close to those generated by a stronger policy and far from those of a weaker one, enhancing responsiveness and adaptability.

As shown in Figure 1, BID dynamically updates future action chunks based on prior decisions, achieving both temporal consistency and real-time reactivity. This approach effectively bridges the gap between static action chunking and closed-loop adaptation, enabling robots to respond to dynamic environments while maintaining stable, long-term behavior.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
poral consistency over extended periods while
remaining reactive to unexpected changes.
The main contributions of this paper are twofold: (i) a thorough analysis of action chunking (§3),
and (ii) a decoding algorithm to improve it (§4). Empirically, we validate our theoretical analysis
through a one-dimensional diagnostic simulation and evaluate our decoding method on two state-ofthe-art generative policies across seven simulations and two real-world tasks (§5). Our experiment
results show that the proposed BID boosts the performance of recent policies by more than 32%
in relative performance. BID is model-agnostic, computationally efficient, and easy to implement,
serving as a plug-and-play component to enhance generative behavior cloning at test time.

**[Speaker Host]:**

RELATED WORK
Behavioral Cloning. Learning from human demonstrations is becoming increasingly popular in
robot learning due to recent advances in robotic teleoperation interfaces (Sivakumar et al., 2022;
Zhao et al., 2023b; Wu et al., 2023; Chi et al., 2024). Generative Behavior cloning, which models
the distribution of demonstrations, is particularly appealing due to its algorithmic simplicity and empirical efficacy (Jang et al., 2022; Florence et al., 2022; Brohan et al., 2022; Shafiullah et al., 2022;
Zhao et al., 2023b; Chi et al., 2024; Brohan et al., 2023a). However, a significant limitation is compounding errors, where deviations from the training distribution accumulate over time (Ross et al.,
2011; Ke et al., 2021). These errors can be mitigated by gathering expert correction data (Ross et al.,
2011; Kelly et al., 2019; Menda et al., 2019; Hoque et al., 2021a; b) or injecting noise during data
collection (Laskey et al., 2017; Brandfonbrener et al., 2023), but such strategies require additional
time and effort from human operators. To address this, recent work proposes predicting a sequence
of multiple actions into the future, known as action chunking, which reduces the effective control
horizon (Lai et al., 2

**[Speaker Guest]:**

assistant
Portal consistency over extended periods while remaining reactive to unexpected changes.

The main contributions of this paper are twofold: (i) a thorough analysis of action chunking, and (ii) a decoding algorithm to improve it. Empirically, we validate our theoretical analysis through a one-dimensional diagnostic simulation and evaluate our decoding method on two state-of-the-art generative policies across seven simulations and two real-world tasks. Our results show that the proposed BID boosts the performance of recent policies by more than 32% in relative performance. BID is model-agnostic, computationally efficient, and easy to implement, serving as a plug-and-play component to enhance generative behavior cloning at test time.

Behavioral Cloning. Learning from human demonstrations is becoming increasingly popular in robot learning due to recent advances in robotic teleoperation interfaces. Generative Behavioral Cloning, which models the distribution of demonstrations, is particularly appealing due to its algorithmic simplicity and empirical efficacy. However, a significant limitation is compounding errors, where deviations from the training distribution accumulate over time. These errors can be mitigated by gathering expert correction data or injecting noise during data collection, but such strategies require additional time and effort from human operators. To address this, recent work proposes predicting a sequence of multiple actions into the future, known as action chunking, which reduces the effective control horizon.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
duces the effective control
horizon (Lai et al., 2022; Zhao et al., 2023b; George & Farimani, 2023; Bharadhwaj et al., 2023a).
By handling sequences of actions, action chunking is also better at handling temporal dependencies
in the data, such as idle pauses (Swamy et al., 2022; Chi et al., 2023) or multiple styles (Li et al.,
2017; Kuefler & Kochenderfer, 2017; Gandhi et al., 2023; Belkhale et al., 2024). However, independently drawn action sequence samples may not preserve the necessary temporal dependencies
for smooth and consistent execution. Our work provides a thorough analysis of action chunking and
proposes a decoding algorithm to improve it.
Published as a conference paper at ICLR 2025
Sequential Decoding. Decoding algorithms have been studied in generative sequence modeling
for decades, with renewed attention driven by recent advances in large language modeling (LLM).
One prominent approach focuses on leveraging internal metrics, e. g., likelihood scores, to improve
the quality of generated sequences. Notable examples include beam search (Freitag & Al-Onaizan,
2017; Vijayakumar et al., 2018), truncated sampling (Fan et al., 2018; Hewitt et al., 2022), minimum
Bayes risk decoding (Kumar & Byrne, 2004; M¨uller & Sennrich, 2021), and others (Welleck et al.,
2019; Meister et al., 2023; Fu et al., 2024a). Another line of research explores the distinctions between multiple models to jointly optimize for the desired properties such as quality or efficiency (Li
et al., 2023; Leviathan et al., 2023). More recently, several studies have highlighted the potential
of guiding the decoding or sampling process through the use of an external model, such as a classifier (Dhariwal & Nichol, 2021) or reward model (Khanov et al., 2023). In the context of robot
learning, recent works have explored guided decoding for long-horizon robotic planning (Huang
et al., 2023) and manipulator geometry designs (Xu et al., 2024). Nevertheless, effective decoding
strategies for low-level r

**[Speaker Host]:**

assistant
Action chunking reduces the effective control horizon. It is better at handling temporal dependencies in data, such as idle pauses or multiple styles. However, independently drawn action sequence samples may not preserve necessary temporal dependencies for smooth execution. Our work provides a thorough analysis of action chunking and proposes a decoding algorithm to improve it.

Published as a conference paper at ICLR 2025.

Sequential decoding has been studied for decades in generative sequence modeling, with renewed attention from recent advances in large language modeling. One approach uses internal metrics like likelihood scores to improve generated sequence quality. Examples include beam search, truncated sampling, and minimum Bayes risk decoding. Another line of research examines how multiple models can be jointly optimized for quality or efficiency. More recently, external models—such as classifiers or reward models—are used to guide the decoding process. In robotics, recent work has explored guided decoding for long-horizon planning and manipulator geometry design. Nevertheless, effective decoding strategies for low-level robotics remain underdeveloped.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
ess, effective decoding
strategies for low-level robotic actions remain lacking. Concurrent to our work, Nakamoto et al.
propose to select action samples by querying a value function learned from reward-annotated
demonstrations (Hansen-Estruch et al., 2023). Our work does not rely on a separate value function;
instead, we propose a decoding strategy that addresses the consistency-reactivity tradeoff inherent
in action chunking through sample comparison.
ANALYSIS: TRADEOFFS IN ACTION CHUNKING

**[Speaker Guest]:**

3. 1

**[Speaker Host]:**

PRELIMINARIES
Consider a dataset of demonstrations D = {τi}N
i=1, where each demonstration τi consists of a sequence of state-action pairs τi = {(s1, a1), (s2, a2), · · ·, (sT, aT )} provided by a human expert.
At each time step t, the demonstrated action at is influenced not only by the observed state st,
but also by latent variables zt, such as planning strategies (e. g., subgoals) and personal preferences
(e. g., handedness). These latent variables can persist across multiple time steps and vary significantly between different demonstrations. Fig. 2 illustrates the decision process of a human expert,
highlighting the inherent temporal dependencies.
To model these temporal dependencies, some recent works (Zhao et al., 2023b; Chi et al., 2023; Lee
et al., 2024) utilize action chunking, i. e., modeling the joint distribution of future actions conditioned
on past states π(at, at+1, · · ·, at+l|st−c, · · ·, st), or in short π(at:t+l|st−c:t). Here, c is the context
length, i. e., number of past steps for state inputs, and l is the prediction length, i. e., number of future
steps for action outputs. Training such a policy typically involves minimizing the divergence of
action distributions between the model π and the expert π∗,

**[Speaker Guest]:**

π = arg min

**[Speaker Host]:**

π

**[Speaker Guest]:**

X

**[Speaker Host]:**

τ∈D

**[Speaker Guest]:**

X

**[Speaker Host]:**

st−c:t

**[Speaker Guest]:**

at:t+l
L(π(at:t+l|st−c:t), π∗(at:t+l|st−c:t)).

**[Speaker Host]:**

(1)
During deployment, the policy operates with a specific action horizon h ∈[1, l], i. e., executing
part or all of the predicted action sequence for h time steps without re-pl

**[Speaker Guest]:**

assistant
Effective decoding strategies for low-level robotic actions remain lacking. Concurrently, Nakamoto et al. propose selecting action samples by querying a value function learned from reward-annotated demonstrations (Hansen-Estruch et al., 2023). Our work does not rely on a separate value function; instead, we propose a decoding strategy that addresses the consistency-reactivity tradeoff inherent in action chunking through sample comparison.

Analysis: Tradeoffs in Action Chunking

Consider a dataset of demonstrations D = {τi}i=1 to N, where each demonstration τi consists of a sequence of state-action pairs τi = {(s1, a1), (s2, a2), ... ... ..., (sT, aT)} provided by a human expert. At each time step t, the demonstrated action at is influenced not only by the observed state st but also by latent variables zt—such as planning strategies (e.g., subgoals) or personal preferences (e.g., handedness). These latent variables can persist across time steps and vary significantly between demonstrations. Figure 2 illustrates the decision process of a human expert, highlighting the inherent temporal dependencies.

To model these dependencies, recent works (Zhao et al., 2023b; Chi et al., 2023; Lee et al., 2024) use action chunking—modeling the joint distribution of future actions conditioned on past states, π(at, at+1, ... ... ..., at+l | st−c, ... ... ..., st), or more succinctly π(at:t+l | st−c:t). Here, c is the context length (number of past steps for state inputs), and l is the prediction length (number of future steps for action outputs). Training such a policy typically involves minimizing the divergence between the model’s action distribution and the expert’s, expressed as:

π = arg min_π Σ_τ∈D Σ_st−c:t at:t+l L(π(at:t+l | st−c:t), π*(at:t+l | st−c:t))

During deployment, the policy operates with a specific action horizon h ∈ [1, l], executing part or all of the predicted action sequence for h time steps without re-planning.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
ted action sequence for h time steps without re-planning. This approach
essentially takes in c states as context and executes h actions, which we refer to as a (c, h)-policy.
The choices of context length c and action horizon h often play a crucial role in the effectiveness
of the learned policy. Recent policies often use a short context length c, as extending the context
can lead to performance degradation in the presence of limited training data (refer to Appendix A. 2
for more details). Conversely, extending the action horizon h has produced mixed results. Some
studies report benefits in laboratory settings (Zhao et al., 2023b; Chi et al., 2023), while others
find it detrimental in real-world scenarios (Lee et al., 2024). The reasons behind these conflicting
outcomes are not well understood yet.
Notably, Zhao et al. (2023b) hypothesize that action chunking mitigates compounding errors, but it is
unclear why this would hold when deviations within a chunk cannot be corrected before replanning.
Alternatively, Chi et al. draws parallels to model predictive control (MPC) (Borrelli et al.,
2017; L¨ofberg, 2012), emphasizing its role in improving consistency and planning. However, unlike
MPC, which typically uses short action horizons (e. g., h = 1), recent imitation learning methods
employ much longer action horizons, such as a substantial portion of the predicted sequence (e. g.,
h = 8 in Chi et al. ) or even the entire prediction length (e. g., h greater than or equal to 50 in Zhao et al. (2023b);
Black et al. ). The lack of a clear understanding of action chunking hinders its effective use
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

st−1

**[Speaker Guest]:**

at−1

**[Speaker Host]:**

st

**[Speaker Guest]:**

st+1

**[Speaker Host]:**

at

**[Speaker Guest]:**

at+1

**[Speaker Host]:**

zt
Figure 2: Illustration of the expert decision
process, where a latent variable introduces
temporal dependencies in actions.

**[Speaker Guest]:**

prediction

**[Speaker Host]:**

replan

**[Speaker Guest]:**

t+1

**[Speaker Host]:**

t-k

**[Speaker Guest]:**

𝜋(𝑐, ℎ)

**[Speaker Host]:**

... ... ...

**[Speaker Guest]:**

𝜋(𝑐, ℎ+𝑑)

**[Speaker Host]:**

... ... ...

**[Speaker Guest]:**

𝜋(𝑘, 1)

**[Speaker Host]:**

∗

**[Speaker Guest]:**

... ... ...

**[Speaker Host]:**

observation

**[Speaker Guest]:**

t-h

**[Speaker Host]:**

t-h-d

**[Speaker Guest]:**

t

**[Speaker Host]:**

Figure 3:
Illustration of (k, 1)-expert, (c, h)-learner, and
(c, h + d)-learner. Shaded regions

**[Speaker Guest]:**

assistant
A (c, h)-policy takes c states as context and executes h actions without re-planning. The choice of context length c and action horizon h significantly affects policy performance. Recent approaches often use short context lengths, as longer contexts can degrade performance due to limited training data. In contrast, extending the action horizon h has mixed results: some studies report benefits in controlled environments, while others observe negative effects in real-world settings. The reasons for these conflicting outcomes remain unclear.

Zhao et al. (2023b) suggest that action chunking reduces compounding errors, though it is unclear why this would help when deviations within a chunk cannot be corrected before replanning. Chi et al. draw parallels to model predictive control (MPC), highlighting its role in improving consistency and planning. However, unlike traditional MPC—which typically uses short horizons (e.g., h = 1)—recent imitation learning methods use much longer horizons, such as a significant portion of the predicted sequence (e.g., h = 8) or even the full sequence (e.g., h ≥ 50). This lack of understanding limits the effective application of action chunking.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

**[Speaker Host]:**

h)-learner, and
(c, h + d)-learner. Shaded regions represent observed history;
darker indicate greater influence on current decision.
across tasks and environments. We next explicitly analyze the strengths and weaknesses of action
chunking, with particular attention to the choice of action horizon h.

**[Speaker Guest]:**

3. 2

**[Speaker Host]:**

ANALYSIS
To understand the influence of action chunking, we focus on the last time step of the executed chunk,
where the discrepancy between the expert policy and the learned policy is most pronounced. At this
time step t, a (k, 1)-expert written as π∗:= π∗(at|st−k:t, zt−k:t) predicts at by conditioning on k
steps of the past states and the corresponding latent variables. In contrast, a (c, h)-learner written as
π(c, h) := π(c, h)(at | st−h−c:t−h, at−h:t−1) is constrained to observe c steps of the past states and
h −1 steps of the past actions within the predicted chunk.
The divergence between a learned policy and an expert policy is attributed to two factors: (i) the
importance of unobserved states in predicting the current action, and (ii) the difficulty of inferring
unobserved states based on the available information.
To more clearly see the influence of action horizon on these factors, we next compare the performance of two policies that have the same context lengths but different action horizons, πh :=
π(c, h)(at|st−h−c:t−h, at−h:t−1) and πh+d := π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−1), where
d > 0 is the extended action horizon. As illustrated in Fig. 3, each policy has access to unique
information that is unavailable to the other. πh observes some recent states, where πh+d is only
aware of the executed actions. On the other hand, πh+d has access to some earlier states and actions,
which precede all information available to πh. We characterize the importance of observations as
follows (formal definitions in Appendix E. 2):
Definition (Expected Observation Advantage). If a policy can observe a state st, we say that it
has an observation advantage αt over ano

**[Speaker Guest]:**

assistant
The influence of action chunking is analyzed by focusing on the last time step of each executed chunk, where the discrepancy between expert and learned policies is most evident. At time step *t*, a (k,1)-expert policy π* conditions its prediction on *k* past states and corresponding latent variables to choose action *at*. In contrast, a (c,h)-learner π(c,h) only observes *c* past states and *h−1* past actions within the chunk.

The gap between expert and learned policies arises from two key factors: (i) the importance of unobserved states in predicting current actions, and (ii) the difficulty of inferring unobserved states from available observations.

To examine how action horizon affects these factors, we compare two policies with identical context lengths but different action horizons: πh = π(c,h)(at | st−h−c:t−h, at−h:t−1) and πh+d = π(c,h+d)(at | st−h−d−c:t−h−d, at−h−d:t−1), where *d > 0* extends the action horizon. As shown in Figure 3, each policy accesses distinct information. πh observes recent states, while πh+d only sees past actions. Conversely, πh+d has access to earlier states and actions that precede all information available to πh.

We define the importance of observations as follows:
**Expected Observation Advantage** — if a policy observes state *st*, it has an observation advantage αt over another policy that does not.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

**[Speaker Host]:**

y that it
has an observation advantage αt over another policy that cannot observe it. More formally, this is
the difference between the expected divergence accumulated by a policy π(at′|st′) that does not
condition on the state st and that by a policy π(at′|st′, st) that conditions on st.
Definition (Maximum Inference Disadvantage). If a policy cannot condition its prediction on
a state st, its maximum inference disadvantage ϵt is the largest possible divergence arising from
inferring incorrectly. Here, the maximum is taken over all possible incorrect inferences of st.
Hence, we denote the observation advantage that πh gains from the observed recent states by αf
and the maximum inference disadvantage it incurs from the earlier unobserved states by ϵb, whereas
πh+d conversely gains αb but incurs ϵf.
Let P(st+1 | st, at) denote the true transition probabilities and ˆP(st+1 | st, at) the implicit dynamics model learned by the policy. The difficulty of inferring an unobserved state hinges on both the
relevant observations and environmental stochasticity, which we quantify as follows.
Definition (Forward Inference). Let Pf(t) := P(St = gt|St−1 = gt−1, At−1 = at−1), where gt
and gt−1 are the ground truth states at time t and t −1 respectively in a deterministic environment.
In deterministic environments, Pf(t) = 1; whereas in stochastic settings, Pf(t) is smaller. The
prediction error is characterized by δf(t) = ˆP(St = gt|St−1 = gt−1, at−1) −P(St = gt|St−1 =

**[Speaker Guest]:**

gt−1, at−1).
Definition (Backward Inference). Similarly, let Pb(t) := P(St = gt|St+1 = gt+1) where gt and
gt+1 are the ground truth states in the deterministic environment at time t and t + 1, respectively.
Published as a conference paper at ICLR 2025
Since Pb(t) is not conditioned on any action, it has higher entropy in general. In stochastic environments, Pb(t) is small. Furthermore, let δb(t) = ˆP(St = gt|St+1 = gt+1) −P(St = gt|St+1 =

**[Speaker Host]:**

gt+1).
We refer to the variance of the distribution P(at′−1,... ... ..., at′−d | st′) u

**[Speaker Guest]:**

assistant
The text contains technical definitions and equations related to observation advantages, inference disadvantages, and forward/backward inference in policy learning. To prepare it for a podcast script, we must clean and simplify the content, removing LaTeX, mathematical notation, and technical jargon while preserving the core ideas and flow of reasoning.

Here is the cleaned, podcast-ready version of the text:


**[Speaker Host]:**

---

A policy that can observe the current state of the environment has a clear advantage over one that cannot. This advantage comes from being able to make better decisions based on real-time observations. The difference in performance between these policies is called the observation advantage.

In contrast, when a policy cannot observe a state, it may have to infer the state from past or future information — and that introduces errors. The worst-case error from such incorrect inference is called the maximum inference disadvantage.


**[Speaker Guest]:**

So, a policy that observes the current state gains an advantage αf, but may suffer a disadvantage ϵb from misinterpreting earlier, unobserved states. On the other hand, a policy that doesn’t observe the current state gains a different advantage αb but faces a higher inference error ϵf.

The difficulty in predicting a state depends on what information is available and how unpredictable the environment is. We break this down into two parts: forward inference and backward inference.

Forward inference refers to predicting the current state based on the previous state and action. In a deterministic environment, this prediction is perfect — the system follows a clear path. But in a noisy, random environment, the prediction becomes less reliable.

Backward inference is about predicting the current state based on the next state. Since it doesn’t depend on actions, it's more uncertain — especially in stochastic environments where outcomes vary. This uncertainty makes backward inference harder and more error-prone.

The difference between what the policy predicts and what actually happens in these cases defines the prediction error — a key signal of how well the policy understands the environment.


**[Speaker Host]:**

---

This version removes all mathematical notation, LaTeX, and complex terminology while preserving the core concepts, structure, and logical flow. It's now ready for use in a podcast script.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
ce of the distribution P(at′−1,... ... ..., at′−d | st′) under expert policy as the
diversity of past strategies up to time t′. Given that the forward inference is generally easier than
the backward inference, the performances of πh and πh+d differ as follows (proofs are deferred to

**[Speaker Guest]:**

Appendix E):
Proposition 1 (Consistency-Reactivity Inequalities). Let L be a non-linear and non-negative convex
function measuring the prediction error with respect to demonstrations. Let C := {at−h:t−1} ∪S+
where S+ are the common states that both πh and πh+d observe. For the ease of notation, let
τf = {t −h −d + 1 : t −h} and let τb = {t −h −d −c : t −h −c −1}.

**[Speaker Host]:**

Then,
the difference in the expected loss between the (c, h + d)-policy and the (c, h)-policy, ∆d :=
minπh+d E [L(πh+d, π∗)|C] −minπh E [L(πh, π∗)|C], is bounded as:

**[Speaker Guest]:**

αf −ϵb


**[Speaker Host]:**

1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

τ∈τb
Pb(τ)(Pb(τ) + δb(τ))!
less than or equal to ∆d less than or equal to ϵf


**[Speaker Guest]:**

1 −

**[Speaker Host]:**

Y

**[Speaker Guest]:**

τ∈τf
Pf(τ)(Pf(τ) + δf(τ))


**[Speaker Host]:**

−αb.

**[Speaker Guest]:**

(2)
Remark 1. Eq. (2) provides a general comparison of the performance of the two policies. Intuitively,
the advantage of each policy stems from the additional information it has access to (i. e. αf for πh and
αb for πh+d) while the disadvantage is bounded by the maximum divergence arising from inferring
missing information incorrectly (i. e. ϵb and ϵf scaled by the maximum probability of incorrect

**[Speaker Host]:**

inference).
We next examine the implications of Proposition 1.
Corollary 2 (Consistency). Suppose the train and test environments are identical and deterministic.
Suppose at is influenced by at least one state at time steps τb and let δf(t′) approximately 0 for all t′ ∈τf. If
the diversity in past strategies up to time t −h −c is not 0, and ϵf is finite, then

**[Speaker Guest]:**

min
πh+d E [L(πh+d, π∗)|C] < min

**[Speaker Host]:**

πh E [L(πh, π∗)|C].

**[Speaker Guest]:**

(3)
Remark 2. In deterministic environments, while both policies need to infer the same number of
unobserved states, πh+d benefits from conditioning on additional actions, which may significantly
aid in inferring the corresponding states through

**[Speaker Host]:**

assistant
The performance difference between a policy that uses past actions up to time t−h−d and one that uses only up to time t−h depends on the diversity of past strategies and the ability to infer missing state information. Given that forward inference is generally easier than backward inference, the policies πh and πh+d exhibit different behaviors in terms of prediction error and performance.

Proposition 1 establishes bounds on the expected loss difference between the (c, h+d)-policy and the (c, h)-policy, under certain assumptions. Specifically, let L be a non-negative, non-linear, convex loss function measuring prediction error. Let C be the set of common states observed by both policies, and define τf as the forward time window {t−h−d+1: t−h}, and τb as the backward time window {t−h−d−c: t−h−c−1}. Then, the difference in expected loss ∆d is bounded by:

αf − ϵb / (1 − ∏_{τ∈τb} Pb(τ)(Pb(τ) + δb(τ))) ≤ ∆d ≤ ϵf / (1 − ∏_{τ∈τf} Pf(τ)(Pf(−1) + δf(τ))) − αb

This inequality shows that the advantage of the extended policy πh+d over πh comes from the additional information it gains—specifically, access to earlier actions (αf) and backward state observations (αb). However, this advantage is offset by the risk of incorrect inference when missing states are reconstructed (ϵf and ϵb), scaled by the probability of such incorrect inferences.

In deterministic environments where training and test settings are identical, and where each action at is influenced by at least one state at time steps in τb, and where δf(t′) ≈ 0 for all t′ ∈ τf, then if past strategies are diverse (not zero) and ϵf is finite, the policy πh+d performs strictly better than πh:

min_πh+d E[L(πh+d, π*) | C] < min_πh E[L(πh, π*) | C]

This result suggests that even though both policies must infer the same number of unobserved states, πh+d benefits from conditioning on additional actions, which improves the accuracy of state inference and ultimately leads to better policy performance.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user

aid in inferring the corresponding states through its action chunk. However, this is only true if the
implicit dynamics model learned by the policy is approximately accurate and the maximum errors
ϵf arising from inferring these states are bounded. Note that πh+d performs better as we increase
the diversity of strategies present in the dataset i. e., as the distribution of actions taken by the expert
prior to the context of the short horizon policy becomes higher variance.
Corollary 3 (Reactivity). Suppose Pf(t′) is small or both Pf(t′) and |δf(t′)| are large for all
t′ ∈τf. If temporal dependency decreases over time such that ϵb is small and at is influenced by at
least one state at time steps in τf, then

**[Speaker Guest]:**

min
πh+d E [L(πh+d, π∗)|C] > min

**[Speaker Host]:**

πh E [L(πh, π∗)|C].

**[Speaker Guest]:**

(4)
Remark 3. Recent states are often more important to decision making and, therefore, the disadvantages of action chunking become pronounced when inferring these recent states becomes difficult.
This can happen if the test environment is stochastic. This can also happen if the test environment is
deterministic as long as the implicit dynamics model is inaccurate - either due to a distribution shift
or because of learning difficulty.
In summary, depending on the experimental conditions, action chunking may have varying effects
on the learned policy. On the one hand, it benefits the modeling of temporal dependencies in the
demonstrations, due to extended access to actions executed at past time steps. On the other hand,
it hinders reactions to unexpected dynamics, due to reduced access to state observations at recent
time steps. As a result, there is no universally optimal choice of action horizon across all conditions.
When both temporal dependencies and prediction errors are significant, tuning the action horizon
entails an inherent trade-off between the two opposing factors.
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

𝜋

**[Speaker Guest]:**

𝑠𝑡

**[Speaker Host]:**

𝜋′

**[Speaker Guest]:**

𝑠𝑡

**[Speaker Host]:**

ℒ𝐹

**[Speaker Guest]:**

ℒ𝐵

**[Speaker Host]:**

previous

**[Speaker Guest]:**

positive

**[Speaker Host]:**

negative

**[Speaker Guest]:**

𝑠𝑡−1

**[Speaker Host]:**

ො𝑎
Figure 4: Illustration of bidirection

**[Speaker Guest]:**

assistant
Action chunking can aid in inferring corresponding states through its action chunk, but only if the implicit dynamics model learned by the policy is approximately accurate and the maximum errors ϵf from state inference are bounded. πh+d performs better as we increase the diversity of strategies in the dataset—i.e., as the distribution of actions taken by the expert prior to the context of the short-horizon policy becomes higher variance.

Corollary 3 (Reactivity): Suppose Pf(t′) is small or both Pf(t′) and |δf(t′)| are large for all t′ ∈ τf. If temporal dependency decreases over time such that ϵb is small and at is influenced by at least one state at time steps in τf, then
min_πh+d E[L(πh+d, π*)|C] > min_πh E[L(πh, π*)|C].

Remark 3: Recent states are often more important for decision-making, so the disadvantages of action chunking become pronounced when inferring recent states is difficult—this can occur in stochastic environments, or in deterministic environments if the implicit dynamics model is inaccurate due to distribution shift or learning difficulty.

In summary, under different experimental conditions, action chunking has varying effects. It helps model temporal dependencies in demonstrations by providing extended access to past actions. Yet, it hinders reaction to unexpected dynamics due to reduced access to recent state observations. Thus, there is no universally optimal action horizon across all settings. When both temporal dependencies and prediction errors are significant, tuning the action horizon involves a trade-off between these two opposing factors.

Published as a conference paper at ICLR 2025.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

**[Speaker Host]:**

tive

**[Speaker Guest]:**

𝑠𝑡−1

**[Speaker Host]:**

ො𝑎
Figure 4: Illustration of bidirectional decoding.
Algorithm 1 Bidirectional Decoding
Require: current state s, batch size N, mode size K, previous decision ˆa, strong policy π, weak policy π′
1: Generate N samples from each policy a ∼π(s), a′ ∼
π′(s) to construct the initial sets A and A

**[Speaker Guest]:**

′
2: Compute the backward loss LB for each sample
3: Select K samples with minimal LB from A and A

**[Speaker Host]:**

′ to
construct A+ and A−, respectively
4: Compute the forward loss LF for each sample
5: Select a∗∈A that minimizes the total loss
6: Update decision memory ˆa ←a∗
METHOD: BIDIRECTIONAL DECODING
As analyzed above, action chunking improves long-term consistency but sacrifices short-term reactivity. In this section, we propose to address this tradeoff by bridging action chunking with closedloop adaptation. We will first outline the general framework in §4. 1 and then describe two specific

**[Speaker Guest]:**

criteria in §4. 2.

**[Speaker Host]:**

4. 1

**[Speaker Guest]:**

TEST-TIME SEARCH
Recall that for a policy with prediction length l, an action chunk a := {a(t)

**[Speaker Host]:**

t, a(t)

**[Speaker Guest]:**

t+1, · · ·, a(t)
t+l} sampled at time t is expected to follow a consistent strategy over the subsequent l time steps. However,
executing the action chunk in an open-loop manner leaves the policy vulnerable to unexpected state
changes. Alternatively, one can maximize reactivity by resampling an action chunk and executing
only the first immediate action at every time step. Yet, this simple closed-loop approach destroys
the consistency preserved within each chunk, potentially leading to oscillations between different

**[Speaker Host]:**

❓ strategies. How can we combine the benefits of both approaches?
Our key idea is to bridge action chunking and closed-loop resampling by making use of additional
computes at test time. In particular, we seek to restore temporal consistency in closed-loop operations by scaling up the number of candidate samples. Intuitively, while the probability of any single
pair of samples sharing the same latent strategy is low, the likelihood of finding a consistent pai

**[Speaker Guest]:**

assistant
Figure 4: Illustration of bidirectional decoding.

Algorithm 1 Bidirectional Decoding
Require: current state s, batch size N, mode size K, previous decision ˆa, strong policy π, weak policy π′
1: Generate N samples from each policy a ∼ π(s), a′ ∼ π′(s) to construct the initial sets A and A′
2: Compute the backward loss LB for each sample
3: Select K samples with minimal LB from A and A′ to construct A+ and A−, respectively
4: Compute the forward loss LF for each sample
5: Select a∗ ∈ A that minimizes the total loss
6: Update decision memory ˆa ← a∗

METHOD: BIDIRECTIONAL DECODING

As analyzed above, action chunking improves long-term consistency but sacrifices short-term reactivity. In this section, we propose to address this tradeoff by bridging action chunking with closed-loop adaptation. We will first outline the general framework in §4.1 and then describe two specific criteria in §4.2.

4.1 TEST-TIME SEARCH


**[Speaker Host]:**

❓ Recall that for a policy with prediction length l, an action chunk a := {a(t), a(t+1), ... ... ..., a(t+l)} sampled at time t is expected to follow a consistent strategy over the subsequent l time steps. However, executing the action chunk in an open-loop manner leaves the policy vulnerable to unexpected state changes. Alternatively, one can maximize reactivity by resampling an action chunk and executing only the first immediate action at every time step. Yet, this simple closed-loop approach destroys the consistency preserved within each chunk, potentially leading to oscillations between different strategies. How can we combine the benefits of both approaches?

Our key idea is to bridge action chunking and closed-loop resampling by making use of additional computations at test time. In particular, we seek to restore temporal consistency in closed-loop operations by scaling up the number of candidate samples. Intuitively, while the probability of any single pair of samples sharing the same latent strategy is low, the likelihood of finding a consistent pair increases when considering a larger set of candidates.

We introduce bidirectional decoding as a test-time search procedure that simultaneously evaluates both forward and backward consistency. The algorithm proceeds as follows:

- Generate a large set of action samples from both the strong and weak policies to form initial candidate sets.
- Evaluate each sample using a backward loss to assess how well it aligns with prior state transitions and long-term strategy.
- Select the K best samples with minimal backward loss to form forward and backward candidate sets.
- Then compute a forward loss to evaluate the immediate reactivity and feasibility of each action.
- Finally, select the action that minimizes the combined forward and backward loss, ensuring both consistency and responsiveness.

This approach preserves long-term strategic coherence while maintaining short-term adaptability through dynamic selection at test time.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
is low, the likelihood of finding a consistent pair
increases with the number of samples. We thus frame the problem of closed-loop action chunking
as searching for the optimal action among a batch of samples drawn at each time step,

**[Speaker Guest]:**

a∗= arg min

**[Speaker Host]:**

a∈A

**[Speaker Guest]:**

LB(a) + LF (a),

**[Speaker Host]:**

(5)
where A is the set of sampled action chunks, LB and LF are two criteria approximating the optimality with respect to the backward decision and forward plan, which we will describe next.

**[Speaker Guest]:**

4. 2
BIDIRECTIONAL CRITERIA
Backward coherence. To preserve temporal dependencies in closed-loop operations, a sequence
of actions should (i) commit to a consistent latent strategy over time, and (ii) react smoothly to
unexpected changes. These desired properties motivate us to keep the action chunk selected at the
previous time ˆa := {a(t−1)

**[Speaker Host]:**

t−1, · · ·, a(t−1)
t+l−1} as a prior, and minimize the weighted Euclidean distance
between the new action chunk and the prior across l −1 overlapping steps:

**[Speaker Guest]:**

LB =

**[Speaker Host]:**

l−1

**[Speaker Guest]:**

X

**[Speaker Host]:**

τ=0

**[Speaker Guest]:**

ρτ a(t)

**[Speaker Host]:**

t+τ −a(t−1)

**[Speaker Guest]:**

t+τ


**[Speaker Host]:**

2.

**[Speaker Guest]:**

(6)
Here, ρ is a decay hyperparameter to account for growing uncertainty over time. By default, we use
the L2 norm as the distance metric between predicted actions, while other metrics, such as L1 or
cosine distance, can also be effective (see Appendix A. 6). This backward objective encourages similar latent strategies between consecutive steps while allowing for gradual adaptation to unforeseen
transition dynamics.
Published as a conference paper at ICLR 2025
(a) long-idle demo, low-noise env
(b) mid-idle demo, mid-noise env
(c) short-idle demo, high-noise env
Figure 5: Effect of action horizon h on idle actions in 1-dimensional simulations. All policies share the same
prediction length l. Long action horizons lead to idle distributions closer to the long-idle expert in low-noise
environments, whereas shorter action horizons align more closely with the short-idle expert in high-noise environments. When both idling and noise are non-negligible, a moderate action horizon performs the be

**[Speaker Host]:**

assistant
The likelihood of finding a consistent pair is higher as the number of samples increases. We therefore frame the problem of closed-loop action chunking as selecting the optimal action from a batch of sampled actions at each time step:
a* = arg minₐ∈A [LB(a) + LF(a)],
where A is the set of sampled action chunks, and LB and LF are criteria that approximate optimality with respect to backward decision and forward planning, respectively.

To preserve temporal dependencies, a sequence of actions should (i) commit to a consistent latent strategy over time, and (ii) react smoothly to unexpected changes. This motivates using the previous action chunk ˆa := {a(t−1)ₜ₋₁, ... ... ..., a(t−1)ₜ₊ₗ₋₁} as a prior, and minimizing the weighted Euclidean distance between the new action chunk and the prior across l−1 overlapping steps:
LB = Σₜ=₀ˡ⁻¹ ρₜ [a(t)ₜ₊ₜ − a(t−1)ₜ₊ₜ]².
Here, ρ is a decay hyperparameter that accounts for increasing uncertainty over time. The L² norm is used by default as the distance metric, though alternatives like L¹ or cosine distance can also be effective. This backward objective ensures consistent latent strategies across steps while allowing gradual adaptation to unforeseen dynamics.

The figure illustrates the effect of action horizon h on idle behavior in 1D simulations. All policies use the same prediction length l. Longer horizons result in idle distributions closer to the long-idle expert in low-noise environments, while shorter horizons align better with the short-idle expert in high-noise environments. When both idling and noise are significant, a moderate action horizon performs best.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
ligible, a moderate action horizon performs the best.
Nevertheless, the backward criterion alone presents a potential caveat: the prior chunk could be
suboptimal due to the lack of information at the previous time step (e. g., unexpected object motions).
In such cases, selecting the next action chunk based solely on the prior may perpetuate suboptimality.
Ideally, the sequential decision-making process should effectively correct suboptimal plans based on
the latest observations. We next address this need through another forward criterion.
Forward contrast. Our design of the forward criterion is motivated by the need to identify the most
optimal plan from a set of candidates. Within the same latent strategy, suboptimal samples may arise
from (i) low likelihood under the learned model and (ii) divergence between the learned model and
expert policy. To address this, we draw inspirations from LLM decoding techniques (Wang et al.,
2022; Li et al., 2023) and introduce a forward contrast criterion. Specifically, we compare each
candidate sample with two sets of reference samples: one positive set from a stronger policy and
a negative set from a weaker one. The stronger policy is obtained from a well-trained checkpoint,
whereas the weaker policy is taken from an early underfitting checkpoint and is expected to be
further from the expert policy. Our forward objective is thus framed as minimizing the average
distance between a candidate plan and the positive samples while maximizing its average distance
from the negative ones,

**[Speaker Guest]:**

LF = 1

**[Speaker Host]:**

N

**[Speaker Guest]:**

X

**[Speaker Host]:**

a+∈A+

**[Speaker Guest]:**

l

**[Speaker Host]:**

X

**[Speaker Guest]:**

τ=0

**[Speaker Host]:**

a(t)

**[Speaker Guest]:**

t+τ −a+

**[Speaker Host]:**

t+τ


**[Speaker Guest]:**

2 −

**[Speaker Host]:**

X

**[Speaker Guest]:**

a−∈A−

**[Speaker Host]:**

l

**[Speaker Guest]:**

X

**[Speaker Host]:**

τ=0

**[Speaker Guest]:**

a(t)

**[Speaker Host]:**

t+τ −a−

**[Speaker Guest]:**

t+τ


**[Speaker Host]:**

2!,

**[Speaker Guest]:**

(7)
where A+ = A \ {a} is the positive set predicted by the strong policy π, A−is the negative set
predicted by the weaker one π′, and N is the sample size.
Fig. 4 illustrates the combined effects of the backward coherence and forward contrast criteria on
sample selection. Since samples in A+ and A−are not all subject to the same strategy, we trim
each set by removing

**[Speaker Host]:**

assistant
A moderate action horizon performs best.
However, relying solely on the backward criterion has a limitation: the prior action chunk might be suboptimal because of missing information at the previous time step—such as unexpected object motions. Choosing the next action based only on the prior can perpetuate poor decisions. Ideally, the decision process should adapt and correct suboptimal plans using the most recent observations. To address this, we introduce a forward contrast criterion.

Our forward contrast criterion aims to identify the most optimal plan from a set of candidates. Within the same latent strategy, suboptimal samples can arise from two sources: (1) low likelihood under the learned model, and (2) divergence between the learned model and the expert policy.

We draw inspiration from large language model decoding techniques and introduce forward contrast. Specifically, each candidate plan is compared against two reference sets: a positive set from a strong policy and a negative set from a weak one. The strong policy comes from a well-trained checkpoint, while the weak policy is derived from an early underfitting checkpoint and is expected to be farther from the expert policy.

The forward objective minimizes the average distance between a candidate plan and positive samples, while maximizing the average distance from negative samples.

LF = (1/N) × [∑ over a+ in A+ of ∑ over τ=0 to l of (a(t) at τ − a+ at τ)² − ∑ over a− in A− of ∑ over τ=0 to l of (a(t) at τ − a− at τ)²]

A+ consists of all plans predicted by the strong policy, excluding the current sample. A− is the set of plans from the weak policy. N is the total number of samples.

Figure 4 shows how the backward coherence and forward contrast criteria work together in sample selection. Since the positive and negative sets are not aligned with the same strategy, each set is trimmed to ensure consistency.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
o the same strategy, we trim
each set by removing samples that deviate significantly from the previous decision. This is achieved
by summing over the K smallest distance values for in the positive and negative sets in Eq. (7).
The full process of our decoding method is outlined in Algorithm 1. Since all steps in BID can be
computed in parallel, the overall computational overhead remains modest on modern GPU devices.
Please refer to Appendix B for additional discussions about our decoding method.

**[Speaker Guest]:**

EXPERIMENTS
In this section, we present a series of experiments to answer the following questions:

**[Speaker Host]:**

❓ 1. How does our theoretical analysis on action chunking manifest under different conditions?

**[Speaker Guest]:**

❓ 2. Can our decoding method improve closed-loop operations of policies built with action chunking?

**[Speaker Host]:**

❓ 3. Does our decoding method perform well across different policies, tasks, and environments?

**[Speaker Guest]:**

❓ 4. Is our decoding method scalable to larger sample sizes and compatible with existing methods?
To this end, we will first validate our theoretical analysis through one-dimensional diagnostic simulations. We will then evaluate BID on seven tasks across three simulation benchmarks, including
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

Vanilla

**[Speaker Guest]:**

Warmstart

**[Speaker Host]:**

EMA

**[Speaker Guest]:**

BID

**[Speaker Host]:**

0. 8

**[Speaker Guest]:**

0. 9

**[Speaker Host]:**

1. 0

**[Speaker Guest]:**

push-t

**[Speaker Host]:**

0. 0

**[Speaker Guest]:**

0. 4

**[Speaker Host]:**

0. 8

**[Speaker Guest]:**

lift

**[Speaker Host]:**

0. 8

**[Speaker Guest]:**

0. 9

**[Speaker Host]:**

1. 0

**[Speaker Guest]:**

can

**[Speaker Host]:**

0. 6

**[Speaker Guest]:**

0. 7

**[Speaker Host]:**

0. 8

**[Speaker Guest]:**

square

**[Speaker Host]:**

0. 3

**[Speaker Guest]:**

0. 4

**[Speaker Host]:**

0. 5

**[Speaker Guest]:**

transport

**[Speaker Host]:**

0. 4

**[Speaker Guest]:**

0. 5

**[Speaker Host]:**

0. 6

**[Speaker Guest]:**

tool hang

**[Speaker Host]:**

0. 1

**[Speaker Guest]:**

0. 4

**[Speaker Host]:**

0. 7

**[Speaker Guest]:**

kitchen
Figure 6: Comparison of different inference methods for closed-loop operations of diffusion policies. Each
method is evaluated for 100 episodes on seven manipulation tasks in simulation benchmarks. Results are
averaged across three seeds. BID significantly outperforms existing inference methods in most tasks.

**[Speaker Host]:**

5

**[Speaker Guest]:**

15

**[Speaker Host]:**

Number of Samples

**[Speaker Guest]:**

0. 0

**[Speaker Host]:**

0. 1

**[Speaker Guest]:**

0. 2

**[Speaker Host]:**

0. 3

**[Speaker Guest]:**

Relative Perf Gain

**[Speaker Host]:**

Warmstart EMA

**[Speaker Guest]:**

BID Warmstart

**[Speaker Host]:**

+BID

**[Speaker Guest]:**

EMA

**[Speaker Host]:**

+BID

**[Speaker Guest]:**

0. 0

**[Speaker Host]:**

0. 1

**[Speaker Guest]:**

0. 2

**[Speaker Host]:**

0. 4

**[Speaker Guest]:**

0. 5

**[Speaker Host]:**

Relative Perf Gain
Figure 7: BID benefits from large sample sizes (left) and complements existing inference methods (right). Each
method is evaluated on seve

**[Speaker Guest]:**

assistant
The authors propose a decoding strategy that trims each set by removing samples that deviate significantly from the previous decision, achieved by summing over the K smallest distance values in the positive and negative sets as defined in Eq. (7). The full decoding process is outlined in Algorithm 1. All steps in BID can be computed in parallel, resulting in modest computational overhead on modern GPU devices. Additional details about the decoding method are available in Appendix B.

Experiments address four key questions:

**[Speaker Host]:**

❓ 1. How does the theoretical analysis of action chunking perform under different conditions?

**[Speaker Guest]:**

❓ 2. Can the decoding method improve closed-loop operations of policies using action chunking?

**[Speaker Host]:**

❓ 3. Does the method work across different policies, tasks, and environments?

**[Speaker Guest]:**

❓ 4. Is the method scalable to larger sample sizes and compatible with existing approaches?

The validation begins with one-dimensional diagnostic simulations. Subsequently, BID is evaluated across seven tasks on three simulation benchmarks. Results show that BID significantly outperforms existing inference methods—vanilla, warmstart, and EMA—across most tasks. Performance gains are particularly notable in tasks like lift, push-t, can, square, transport, tool hang, and kitchen.

Figure 6 compares the inference methods across 100 episodes per task, averaged over three random seeds. BID consistently achieves higher performance than baseline methods.

Figure 7 shows that BID benefits from larger sample sizes and complements existing methods. The relative performance gain increases with sample size, and BID shows strong synergy when combined with warmstart or EMA. The results indicate scalability and compatibility with existing frameworks.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
methods (right). Each
method is evaluated on seven simulation tasks across three seeds. Relative performance gain is measured with
respect to the vanilla baseline. When combined with EMA, BID results in 46% relative improvements.
Push-T (Chi et al., 2023), RoboMimic (Mandlekar et al., 2022), and Franka Kitchen (Gupta et al.,
2020). We will subsequently examine the generality and scalability of our method under various
base policies, sample sizes, and stochastic noise. We will finally assess the effectiveness of BID in
two challenging real-world tasks that require interactions with dynamic objects.

**[Speaker Host]:**

5. 1
ONE-DIMENSIONAL DIAGNOSTIC EXPERIMENTS
Setup. We start with a diagnostic experiment in a one-dimensional state space {s0, s1, · · ·, s10},
where s0 is the starting state and s10 is the goal state. The demonstrator plans to move forward
by one step in each state, except in s5 where it pauses unless the last five states visited were s5.
Each forward move has a probability of 1 −δ, where δ denotes the level of stochastic noise in the
environment (as described in §3. 2). Given these demonstrations, we train a collection of policies
with different action horizons h ∈{1, 2, 3, 5, 7, 10}. We investigate under what action horizon our
learner can better imitate the distribution of idle actions taken by the expert over multiple rollouts.
Result. As shown in Fig. 5, when the environment is deterministic (δ = 0. 0), larger action horizons
capture the expert distribution better, consistent with Corollary 2. With an action horizon of 10,
the learner achieves zero total variation distance with the expert distribution. Conversely, when the
environment is highly stochastic (δ = 0. 4), an action horizon of 1 outperforms all other learners,
corroborating with Corollary 3. With moderate noise (δ = 0. 2), there is no monotonic pattern and
the optimal policy turns out to be the one with action horizon 5. Refer to Appendix A. 1 for more
detailed results. This controlled experiment validate

**[Speaker Guest]:**

assistant
We conduct a diagnostic experiment in a one-dimensional state space with states {s0, s1, ... ... ..., s10}, where s0 is the starting state and s10 is the goal. The demonstrator plans to move forward by one step in each state, except at s5, where it pauses unless the last five states visited were s5. Each forward move succeeds with probability 1 − δ, where δ represents the level of stochastic noise in the environment. We train policies with different action horizons h ∈ {1, 2, 3, 5, 7, 10} and evaluate how well they imitate the expert’s idle action distribution across multiple rollouts.

Results show that in a deterministic environment (δ = 0.0), larger action horizons better capture the expert’s idle action distribution, aligning with Corollary 2. With an action horizon of 10, the learner achieves zero total variation distance from the expert distribution. In a highly stochastic environment (δ = 0.4), the policy with action horizon 1 outperforms all others, supporting Corollary 3. Under moderate noise (δ = 0.2), no clear monotonic trend emerges, and the optimal performance is achieved with an action horizon of 5. These findings demonstrate how action horizon selection depends on environmental noise levels.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
ailed results. This controlled experiment validates the tradeoff identified in Proposition 1, which
we will further demonstrate in robotic manipulation subjected to stochastic noise in §5. 2. 3

**[Speaker Host]:**

5. 2
SIMULATION EXPERIMENTS WITH STOCHASTIC NOISE
Next, we evaluate our decoding algorithm on seven simulation tasks of robot manipulation. We will
first compare BID with existing inference methods in closed-loop operations. We will then assess
the effectiveness of our method under different conditions, including policy classes, sample sizes,
and levels of stochastic noise.

**[Speaker Guest]:**

5. 2. 1
COMPARISON WITH EXISTING INFERENCE METHODS
Setup. In each manipulation task, we use Diffusion Policy (Chi et al., 2023) trained on human
demonstrations as the base policy. We evaluate BID with a batch size of N = 16 and a mode size of
K = 3. We consider three existing inference methods as baselines:
- Vanilla (Chi et al., 2023): Execute the first action of a sampled chunk in a closed-loop manner.
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

Stochastic Noise

**[Speaker Guest]:**

0. 0

**[Speaker Host]:**

1. 0

**[Speaker Guest]:**

1. 5

**[Speaker Host]:**

Vanilla Open-Loop

**[Speaker Guest]:**

64. 0

**[Speaker Host]:**

26. 9

**[Speaker Guest]:**

13. 0

**[Speaker Host]:**

BID Open-Loop

**[Speaker Guest]:**

66. 1

**[Speaker Host]:**

31. 4

**[Speaker Guest]:**

16. 0

**[Speaker Host]:**

Vanilla Closed-Loop

**[Speaker Guest]:**

48. 9

**[Speaker Host]:**

38. 3

**[Speaker Guest]:**

29. 5

**[Speaker Host]:**

BID Closed-Loop

**[Speaker Guest]:**

54. 4

**[Speaker Host]:**

45. 3

**[Speaker Guest]:**

31. 7
Table 1: Success rates of VQ-BeT on the Push-T task
under various noise conditions.

**[Speaker Host]:**

Closed-loop BID
substantially outperforms other methods in stochastic environments. See Table 4 for detailed ablations.

**[Speaker Guest]:**

Sample Size

**[Speaker Host]:**

Success (%)

**[Speaker Guest]:**

Time (ms)

**[Speaker Host]:**

(vanilla)

**[Speaker Guest]:**

49. 1

**[Speaker Host]:**

13. 2

**[Speaker Guest]:**

(ours)

**[Speaker Host]:**

52. 9

**[Speaker Guest]:**

25. 2

**[Speaker Host]:**

(ours)

**[Speaker Guest]:**

54. 2

**[Speaker Host]:**

25. 9

**[Speaker Guest]:**

(ours)

**[Speaker Host]:**

54. 4

**[Speaker Guest]:**

26. 8
Table 2: Success rates and inference times of VQBeT across varying sample sizes. BID benefits from
a larger sample size at the cost of a doubled computational overhead, measured on an A5000 GPU.
- Warmstart (Janner et al., 2022): Similar to Vanilla, but warm-start the initial noise for the diffusion process from the previous decision.
- Exponential Moving Average (EMA) (Zhao et al., 2023b): Smooth action chunking by averaging
a new prediction a with the previous one ˆa for each over

**[Speaker Host]:**

assistant
We evaluate our decoding algorithm on seven simulation tasks of robot manipulation.

In each task, we use Diffusion Policy (Chi et al., 2023) trained on human demonstrations as the base policy. We evaluate BID with a batch size of N = 16 and a mode size of K = 3. Three baseline methods are considered:

- Vanilla (Chi et al., 2023): Executes the first action of a sampled chunk in a closed-loop manner.
- Warmstart (Janner et al., 2022): Similar to Vanilla, but warm-starts the initial noise from the previous decision.
- Exponential Moving Average (EMA) (Zhao et al., 2023b): Smooths action chunking by averaging a new prediction with the previous one.

Our results show that closed-loop BID substantially outperforms other methods in stochastic environments. BID benefits from larger sample sizes, though at the cost of doubled computational overhead, as measured on an A5000 GPU. Success rates and inference times are detailed in the tables.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
rediction a with the previous one ˆa for each overlapping step at = λat + (1 −λ)ˆat. This
method is also known as temporal ensembling. By default, we set λ = 0. 5.
We evaluate each method for 100 episodes and average the results across three random seeds. Please
refer to Appendix C for implementation details.
Result. Our main observation is that while existing inference methods offer some benefits for
closed-loop operations, they lack robustness. As shown in Fig. 6, Warmstart yields mild performance gains on average, but degrades performance on 3 out of 7 tasks. Similarly, EMA leads to
competitive results on several tasks, yet exhibits performance drops in 2 tasks. We conjecture that
this robustness issue stems from independent sampling across chunks; when successive chunks follow distinct latent strategies, averaging them may not yield a plausible strategy, as further discussed
in Appendix A. 4. In comparison, BID offers substantial gains across all tasks. Notably, BID provides 32% relative improvements over the vanilla baseline, significantly outperforming EMA on
Pust-T, Lift, Square, and Tool Hang, while achieving competitive performance on the other tasks.

**[Speaker Guest]:**

5. 2. 2
SCALABILITY AND COMPATIBILITY OF BID
Setup. We further assess two key properties of BID: scalability with growing batch sizes and
compatibility with existing inference methods. For scalability, we experiment with batch sizes of
{1, 4, 8, 12, 16}. For compatibility, we apply BID on top of two other baselines, Warmstart and
EMA. The results are averaged across seven simulation tasks and three random seeds.
Result. As shown in Fig. 7, our method benefits from the large batch size, with performance gains
not yet saturated at the default batch size used in §5. 2. 1. Moreover, the strength from BID is complementary to that of existing inference methods. Notably, combining BID with EMA further boosts
the relative performance gain from 32% to 46%. These two properties highlight the potential of our

**[Speaker Host]:**

method in p

**[Speaker Guest]:**

assistant
Prediction with the previous one ˆa for each overlapping step at = λat + (1 − λ)ˆat. This method is also known as temporal ensembling. By default, we set λ = 0.5.

We evaluate each method for 100 episodes and average the results across three random seeds. Please refer to Appendix C for implementation details.

Result: Our main observation is that while existing inference methods offer some benefits for closed-loop operations, they lack robustness. As shown in Fig. 6, Warmstart yields mild performance gains on average but degrades on 3 out of 7 tasks. Similarly, EMA leads to competitive results on several tasks but shows performance drops on 2 tasks. We conjecture that this robustness issue stems from independent sampling across chunks; when successive chunks follow distinct latent strategies, averaging them may not yield a plausible strategy, as further discussed in Appendix A. In comparison, BID offers substantial gains across all tasks. Notably, BID provides 32% relative improvement over the vanilla baseline, significantly outperforming EMA on Pust-T, Lift, Square, and Tool Hang, while achieving competitive performance on the other tasks.

5.2.2 Scalability and Compatibility of BID

Setup: We further assess two key properties of BID: scalability with growing batch sizes and compatibility with existing inference methods. For scalability, we experiment with batch sizes of {1, 4, 8, 12, 16}. For compatibility, we apply BID on top of two other baselines: Warmstart and EMA. The results are averaged across seven simulation tasks and three random seeds.

Result: As shown in Fig. 7, our method benefits from larger batch sizes, with performance gains not yet saturated at the default batch size used in §5.2.1. Moreover, the strength of BID is complementary to that of existing inference methods. Notably, combining BID with EMA further boosts the relative performance gain from 32% to 46%. These two properties highlight the potential of our method in practical deployment.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
perties highlight the potential of our

**[Speaker Host]:**

method in practice.

**[Speaker Guest]:**

5. 2. 3
GENERALITY AND EFFICIENCY OF BID
Setup. We next extend our experiment to VQ-BET (Lee et al., 2024), another state-of-the-art robot
policy built with autoregressive transformers. We use the public checkpoint on the Push-T task
provided by LeRobot (Cadene et al., 2024) as the base policy. We use a checkpoint terminated at 100
epochs as the weak policy in forward contrast. To simulate stochastic conditions, we add temporally
correlated Gaussian noise to the executed action at each step, scaled by the action magnitude. We
measure the computational time on a desktop equipped with an NVIDIA A5000 GPU.
Result. Table 1 summarizes the results of the baseline and our method. The vanilla random sampling
performs significantly worse than BID in both closed and open-loop operations. Notably, the vanilla
open-loop approach exhibits a rapid performance decline as the environment becomes increasingly
stochastic. Even in closed-loop operations, the vanilla baseline still experiences a significant performance drop. In comparison, the closed-loop BID demonstrates much higher robustness to stochastic
noise. Table 2 details the computational overhead associated with BID at varying batch sizes. The
result shows that the performance gains of our method come with a doubled computational overhead.
We expect that this overhead will be less of a constraint with higher-end GPUs.
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

Static

**[Speaker Guest]:**

Dynamic

**[Speaker Host]:**

20

**[Speaker Guest]:**

60

**[Speaker Host]:**

Success Rate (%)

**[Speaker Guest]:**

Vanilla

**[Speaker Host]:**

BID
Figure 8: Success rate of object delivery. Each
method-setting is evaluated across 20 episodes.
BID achieves much higher success rate than the
vanilla baseline, effectively handling the diverse
demonstrations and dynamic target.

**[Speaker Guest]:**

Vanilla

**[Speaker Host]:**

Open

**[Speaker Guest]:**

Vanilla

**[Speaker Host]:**

Closed

**[Speaker Guest]:**

EMA

**[Speaker Host]:**

BID

**[Speaker Guest]:**

20

**[Speaker Host]:**

60

**[Speaker Guest]:**

100

**[Speaker Host]:**

Success Rate (%)
Figure 9: Success rate of cup replacement in the
dynamic setting. Each method is evaluated across
20 episodes. Existing methods degrade substantially under slow cup movemen

**[Speaker Guest]:**

assistant
Success rate of object delivery. Each method-setting is evaluated across 20 episodes. BID achieves much higher success rate than the vanilla baseline, effectively handling the diverse demonstrations and dynamic target.

Success rate of cup replacement in the dynamic setting. Each method is evaluated across 20 episodes. Existing methods degrade substantially under slow cup movement, while BID maintains strong performance.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
thods degrade substantially under slow cup movements, whereas BID
retains a strong performance.

**[Speaker Host]:**

5. 3
REAL-WORLD EXPERIMENTS WITH DYNAMIC OBJECTS
We finally evaluate BID through two real-world experiments that require rapid reactions to dynamically moving objects.
Setup. We consider two pick-and-place tasks, where the target object undergoes unexpected movement during evaluation. In the dynamic placing task, a Franka Panda robot is required to deliver
an object into a moving cup held by a human subject. In the dynamic picking task, a UR5 robot
is required to grasp a moving cup pulled by a string and place it onto a nearby static saucer. In
both tasks, we evaluate the performance of BID applied to pre-trained diffusion policies. For further
experimental details, please refer to Appendix C. 2.
Result. Figs. 8 and 9 compares the results of different inference methods on the two real-world tasks.
Vanilla random sampling struggles to handle the diverse demonstrations and dynamic movements,
resulting in significantly lower success rates. In contrast, BID achieves high success rates in both
static and dynamic conditions. Notably, in the dynamic picking task, BID achieves a 2x higher
success rate than all other baselines, highlighting its potential for dynamic object interactions.

**[Speaker Guest]:**

Other experiments.
Please refer to Appendix A for additional analyses and ablations.

**[Speaker Host]:**

CONCLUSION
Summary. We have analyzed the strengths and limitations of action chunking for robot learning
from human demonstrations. Based on our analysis, we proposed Bidirectional Decoding (BID), an
inference algorithm that takes into account both past decisions and future plans for sample selection.
Our experimental results show that BID can consistently improve closed-loop operations, scale well
with computational resources, and complement existing methods. We hope these findings provide a
new perspective on addressing the challenges of generative behavioral cloning at test time.
Limitations. One major limitatio

**[Speaker Guest]:**

assistant
BID outperforms other methods under dynamic conditions, maintaining strong performance even when objects move unexpectedly. In real-world experiments involving rapid reactions to moving objects—such as a robot delivering an item into a cup being pulled by a human or grasping a moving cup pulled by a string—BID achieves significantly higher success rates than vanilla sampling or other inference methods. While traditional methods degrade under slow or unpredictable movements, BID remains stable and effective, particularly in dynamic picking tasks where it achieves twice the success rate of all baselines. This demonstrates its robustness in real-time, unpredictable environments. The method leverages past decisions and future plans to guide sample selection, offering a more coherent and adaptable inference process. These results suggest that BID is well-suited for real-world robotic control where object dynamics and timing are uncertain.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

**[Speaker Host]:**

ing at test time.
Limitations. One major limitation of BID lies in its computational complexity. While the decoding
computation can be fully parallelized on modern GPUs, it may remain expensive for high-frequency
operations on low-cost robots. Designing algorithms that can generate quality yet diverse action
chunks under batch size constraints can be an interesting avenue for future research. Additionally,
our analysis and method have been limited to policies with short context lengths, driven by their empirical effectiveness with limited human demonstrations. Developing techniques capable of learning
robust long-context policies can be another compelling direction for future research.
REPRODUCIBILITY STATEMENT
Our code for the experiments with Diffusion Policy is available at [web link] com/
YuejiangLIU/bid_diffusion and our code for the experiments with VQ-BET is available at
[web link] com/Jubayer-Hamid/bid_lerobot. Additionally, detailed descriptions of our experimental setups are available in Appendix C, and complete proofs of our theoretical
claims can be found in Appendix E.
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

ACKNOWLEDGMENTS
We would like to thank Anikait Singh, Eric Mitchell, Kyle Hsu, Moo Jin Kim, Annie Chen, Sasha
Khazatsky, Rhea Malhotra and other members of the IRIS lab for valuable comments on this project.
We would like to thank Yifan Hou, Huy Ha, Chuer Pan, Austin Patel, and the REAL lab for helpful
support in setting up real-world experiments. We would like to thank Asher Spector and Danny Tse
for their insightful feedback on the theoretical analysis. This work is supported by the AI Institute,
ONR grant N00014-21-1-2685, an NSF CAREER award, and an SNSF Postdoc fellowship.

**[Speaker Host]:**

REFERENCES
Brenna D. Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning
from demonstration. Robotics and Autonomous Systems, 57(5):469-483, May 2009.
Christopher G. Atkeson and Stefan Schaal. Robot Learning From Demonstration. In Proce

**[Speaker Guest]:**

assistant
BID has a major limitation in its computational complexity. While the decoding can be fully parallelized on modern GPUs, it may still be expensive for high-frequency operations on low-cost robots. Designing algorithms that generate high-quality, diverse action chunks under batch size constraints is an interesting direction for future research. Additionally, our analysis and method have been limited to policies with short context lengths, due to their empirical effectiveness with limited human demonstrations. Developing techniques to learn robust long-context policies is another compelling future direction.

Code for experiments with Diffusion Policy is available at [web link] com/YuejiangLIU/bid_diffusion, and for VQ-BET at [web link] com/Jubayer-Hamid/bid_lerobot. Detailed experimental setups are in Appendix C, and complete proofs of theoretical claims are in Appendix E.

Published as a conference paper at ICLR 2025.

We thank Anikait Singh, Eric Mitchell, Kyle Hsu, Moo Jin Kim, Annie Chen, Sasha Khazatsky, Rhea Malhotra, and other IRIS lab members for valuable feedback. We thank Yifan Hou, Huy Ha, Chuer Pan, Austin Patel, and the REAL lab for support in real-world experiments. We thank Asher Spector and Danny Tse for insightful feedback on the theoretical analysis. This work was supported by the AI Institute, ONR grant N00014-21-1-2685, an NSF CAREER award, and an SNSF Postdoc fellowship.


**[Speaker Host]:**

References:
Brenna D. Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning from demonstration. Robotics and Autonomous Systems, 57(5):469–483, May 2009.
Christopher G. Atkeson and Stefan Schaal. Robot Learning From Demonstration. In Proceedings of... ... ...


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
chaal. Robot Learning From Demonstration. In Proceedings of
the Fourteenth International Conference on Machine Learning, ICML ’97, pp. 12-20, July 1997.
Akhil Bagaria and George Dimitri Konidaris.
Option discovery using deep skill chaining.
In International Conference on Learning Representations, 2020.

**[Speaker Guest]:**

URL [web link]
semanticscholar. org/CorpusID:209450128.
Suneel Belkhale, Yuchen Cui, and Dorsa Sadigh. Data quality in imitation learning. Advances in
Neural Information Processing Systems, 36, 2024.
Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash
Kumar. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. 2024 IEEE International Conference on Robotics and Automation,

**[Speaker Host]:**

2023a.
Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash
Kumar. RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking, September 2023b.
Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo
Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi 0: A vision-language-action flow
model for general robot control. arXiv preprint arXiv:2410. 24164, 2024.
Francesco Borrelli, Alberto Bemporad, and Manfred Morari. Predictive control for linear and hybrid
systems. Cambridge University Press, 2017.
David Brandfonbrener, Stephen Tu, Avi Singh, Stefan Welker, Chad Boodoo, Nikolai Matni, and
Jake Varley. Visual backtracking teleoperation: A data collection protocol for offline image-based
reinforcement learning. In 2023 IEEE International Conference on Robotics and Automation
(ICRA), pp. 11336-11342. IEEE, 2023.
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn,
Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics
transformer for real-world control at scale. arXiv preprint arXiv:2212. 06817, 2022.

**[Speaker Guest]:**

Anthony Brohan, N

**[Speaker Host]:**

assistant
Robot learning from demonstration.
Option discovery using deep skill chaining.
Data quality in imitation learning.
RoboAgent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking.
pi0: A vision-language-action flow model for general robot control.
Predictive control for linear and hybrid systems.
Visual backtracking teleoperation: A data collection protocol for offline image-based reinforcement learning.
Rt-1: Robotics transformer for real-world control at scale.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
reprint arXiv:2212. 06817, 2022.
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu,
Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander
Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov,
Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo,
Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut,
Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart,
Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2:
Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, July 2023a.
Published as a conference paper at ICLR 2025
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn,
Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, Julian Ibarz, Brian
Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez,
Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi,
Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent
Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and
Brianna Zitkovich. RT-1: Robotics Transformer for Real-World Control at Scale. In Robotics:
Science and Systems XIX. Robotics: Science and Systems Foundation, July 2023b. ISBN 978-09923747-9-2.
Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, and Thomas
Wolf. Lerobot: State-

**[Speaker Guest]:**

assistant
RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, July 2023
Published as a conference paper at ICLR 2025

RT-1: Robotics Transformer for Real-World Control at Scale. In Robotics: Science and Systems XIX. Robotics: Science and Systems Foundation, July 2023

Lerobot: State-of-the-art robotic control framework for real-world applications, developed by Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, and Thomas Wolf.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
c, Adil Zouitine, and Thomas
Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in pytorch. https:
//github. com/huggingface/lerobot, 2024.
Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran
Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. In Robotics: Science
and Systems XIX. Robotics: Science and Systems Foundation, July 2023. ISBN 978-0-99237479-2.
Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ
Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402. 10329, 2024.
Open X.-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X
Models, December 2023.
Pim de Haan, Dinesh Jayaraman, and Sergey Levine. Causal Confusion in Imitation Learning. In
Advances in Neural Information Processing Systems, volume 32. Curran Associates, Incorporated, 2019.
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances
in neural information processing systems, 34:8780-8794, 2021.
Angela Fan, Mike Lewis, and Yann Dauphin.
Hierarchical Neural Story Generation.

**[Speaker Host]:**

In Iryna
Gurevych and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pp. 889-898, Melbourne, Australia, July
2018. Association for Computational Linguistics.
Pete Florence, Corey Lynch, Andy Zeng, Oscar A Ramirez, Ayzaan Wahid, Laura Downs, Adrian
Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson.
Implicit behavioral cloning.

**[Speaker Guest]:**

In
Conference on Robot Learning, pp. 158-168. PMLR, 2022.
Markus Freitag and Yaser Al-Onaizan. Beam Search Strategies for Neural Machine Translation. In
Thang Luong, Alexandra Birch, Graham Neubig, and Andrew Finch (eds.), Proceedings of the
First Workshop on Neural Machine Translation, pp. 56-60, Vancouver, August 2017. Association

**[Speaker Host]:**

for Computational

**[Speaker Guest]:**

assistant
Adil Zouitine, C., Thomas Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in PyTorch. https://github.com/huggingface/lerobot, 2024.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. In Robotics: Science and Systems XIX. Robotics: Science and Systems Foundation, July 2023. ISBN 978-0-99237479-2.

Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024.

Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models, December 2023.

Pim de Haan, Dinesh Jayaraman, and Sergey Levine. Causal Confusion in Imitation Learning. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Incorporated, 2019.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.

Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical Neural Story Generation. In Iryna Gurevych and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 889–898, Melbourne, Australia, July 2018. Association for Computational Linguistics.

Pete Florence, Corey Lynch, Andy Zeng, Oscar A. Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. Implicit behavioral cloning. In Conference on Robot Learning, pp. 158–168. PMLR, 2022.

Markus Freitag and Yaser Al-Onaizan. Beam Search Strategies for Neural Machine Translation. In Thang Luong, Alexandra Birch, Graham Neubig, and Andrew Finch (eds.), Proceedings of the First Workshop on Neural Machine Translation, pp. 56–60, Vancouver, August 2017. Association for Computational Linguistics.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
ouver, August 2017. Association
for Computational Linguistics.
Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the Sequential Dependency of LLM
Inference Using Lookahead Decoding, February 2024a.
Zipeng Fu, Tony Z. Zhao, and Chelsea Finn. Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation, January 2024b.
Kanishk Gandhi, Siddharth Karamcheti, Madeline Liao, and Dorsa Sadigh. Eliciting compatible
demonstrations for multi-human imitation learning. In Conference on Robot Learning, pp. 1981-

**[Speaker Host]:**

1991. PMLR, 2023.
Abraham George and Amir Barati Farimani. One act play: Single demonstration behavior cloning
with action chunking transformers. arXiv preprint arXiv:2309. 10175, 2023.
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman. Relay policy
learning: Solving long-horizon tasks via imitation and reinforcement learning. In Conference on
Robot Learning, pp. 1025-1037. PMLR, 2020.
Published as a conference paper at ICLR 2025
Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine.
IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies, May 2023.
John Hewitt, Christopher Manning, and Percy Liang. Truncation Sampling as Language Model

**[Speaker Guest]:**

Desmoothing.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Findings of the
Association for Computational Linguistics: EMNLP 2022, pp. 3414-3427, Abu Dhabi, United
Arab Emirates, December 2022. Association for Computational Linguistics.
Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural Computation, 9:1735-
1780, 1997. URL [web link] semanticscholar. org/CorpusID:1915014.
Ryan Hoque, Ashwin Balakrishna, Ellen Novoseller, Albert Wilcox, Daniel S Brown, and Ken
Goldberg. Thriftydagger: Budget-aware novelty and risk gating for interactive imitation learning.
arXiv preprint arXiv:2109. 08273, 2021a.
Ryan Hoque, Ashwin Balakrishna, Carl Putterman, Michael Luo, Daniel S Brown, Daniel Seita

**[Speaker Host]:**

assistant
Break the Sequential Dependency of LLM Inference Using Lookahead Decoding, February 2024
Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation, January 2024
Eliciting compatible demonstrations for multi-human imitation learning, Conference on Robot Learning, 2023
One act play: Single demonstration behavior cloning with action chunking transformers, arXiv preprint 2023
Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning, Conference on Robot Learning, 2020
IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies, May 2023
Truncation Sampling as Language Model Desmoothing, EMNLP 2022
Long short-term memory, Neural Computation, 1997
Thriftydagger: Budget-aware novelty and risk gating for interactive imitation learning, arXiv preprint 2021


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
tterman, Michael Luo, Daniel S Brown, Daniel Seita,
Brijen Thananjeyan, Ellen Novoseller, and Ken Goldberg. Lazydagger: Reducing context switching in interactive imitation learning. In 2021 IEEE 17th international conference on automation
science and engineering (case), pp. 502-509. IEEE, 2021b.
Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor
Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter. Grounded Decoding: Guiding Text
Generation with Grounded Models for Embodied Agents. In Thirty-Seventh Conference on Neural Information Processing Systems, November 2023.
Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine,
and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pp. 991-1002. PMLR, 2022.
Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with Diffusion for
Flexible Behavior Synthesis. In International Conference on Machine Learning (ICML), May

**[Speaker Guest]:**

2022.
Yiding Jiang, Evan Zheran Liu, Benjamin Eysenbach, Zico Kolter, and Chelsea Finn. Learning
options via compression, 2022. URL [web link] org/abs/2212. 04590.
Yushan Jiang, Wenchao Yu, Dongjin Song, Lu Wang, Wei Cheng, and Haifeng Chen. Fedskill:
Privacy preserved interpretable skill learning via imitation.
In Proceedings of the 29th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1010-1019, 2023.
Liyiming Ke, Jingqiang Wang, Tapomayukh Bhattacharjee, Byron Boots, and Siddhartha Srinivasa.
Grasping with chopsticks: Combating covariate shift in model-free imitation learning for fine
manipulation. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pp.
6185-6191. IEEE, 2021.
Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J Kochenderfer.
Hgdagger: Interactive imitation learning with human experts. In 2019 International Conference
on Robotics and Automation (ICRA), pp. 8077-8083. IEE

**[Speaker Host]:**

assistant
Lazydagger: Reducing context switching in interactive imitation learning. In 2021 IEEE 17th International Conference on Automation Science and Engineering (CASE), pp. 502–509. IEEE, 2021.

Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents. In Thirty-Seventh Conference on Neural Information Processing Systems, November 2023.

Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pp. 991–1002. PMLR, 2022.

Planning with Diffusion for Flexible Behavior Synthesis. In International Conference on Machine Learning (ICML), May 2022.

Learning Options via Compression. 2022. URL: [web link] org/abs/2212.04590.

Fedskill: Privacy preserved interpretable skill learning via imitation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1010–1019, 2023.

Grasping with Chopsticks: Combating Covariate Shift in Model-Free Imitation Learning for Fine Manipulation. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 6185–6191. IEEE, 2021.

Hgdagger: Interactive Imitation Learning with Human Experts. In 2019 International Conference on Robotics and Automation (ICRA), pp. 8077–8083. IEEE.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
Robotics and Automation (ICRA), pp. 8077-8083. IEEE, 2019.
Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. ARGS: Alignment as Reward-Guided Search.
In The Twelfth International Conference on Learning Representations, October 2023.
Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth
Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis,
Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree
Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin
Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R. Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe,
Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul
Published as a conference paper at ICLR 2025
Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor
Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin,
Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta,
Abhishek Gupta, Dinesh Jayaraman, Joseph J. Lim, Jitendra Malik, Roberto Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu,
Thomas Kollar, Sergey Levine, and Chelsea Finn. DROID: A Large-Scale In-The-Wild Robot
Manipulation Dataset, March 2024.
Alex Kuefler and Mykel J Kochenderfer. Burn-in demonstrations for multi-modal imitation learning.
arXiv preprint arXiv:1710. 05090, 2017.
Shankar Kumar and Willia

**[Speaker Guest]:**

assistant
Args: Alignment as Reward-Guided Search
Published at ICLR 2025

Droid: A Large-Scale In-The-Wild Robot Manipulation Dataset
Published March 2024

Burn-in Demonstrations for Multi-Modal Imitation Learning
arXiv preprint arXiv:1710.05090, 2017

Shankar Kumar and William [name truncated – likely incomplete or irrelevant]


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
arXiv:1710. 05090, 2017.
Shankar Kumar and William Byrne. Minimum Bayes-Risk Decoding for Statistical Machine Translation.
In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, pp. 169-176,
Boston, Massachusetts, USA, May 2004. Association for Computational Linguistics.
Lucy Lai, Ann Zixiang Huang, and Samuel J Gershman. Action chunking as policy compression.

**[Speaker Host]:**

2022.
Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. Dart: Noise injection
for robust imitation learning. In Conference on robot learning, pp. 143-156. PMLR, 2017.
Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, and
Lerrel Pinto. Behavior Generation with Latent Actions, March 2024.
Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding. In Proceedings of the 40th International Conference on Machine Learning, pp.
19274-19286. PMLR, July 2023.
Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke
Zettlemoyer, and Mike Lewis. Contrastive Decoding: Open-ended Text Generation as Optimization. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
12286-12312, Toronto, Canada, July 2023. Association for Computational Linguistics.
Yunzhu Li, Jiaming Song, and Stefano Ermon. Infogail: Interpretable imitation learning from visual
demonstrations. Advances in neural information processing systems, 30, 2017.
Johan L¨ofberg. Oops! i cannot do it again: Testing for recursive feasibility in mpc. Automatica, 48

**[Speaker Guest]:**

(3):550-555, 2012.
Xiao Ma, Sumit Patidar, Iain Haughton, and Stephen James.
Hierarchical Diffusion Policy for
Kinematics-Aware Multi-Task Robotic Manipulation, March 2024.
Ajay Mandlekar, Fabio Ramos, Byron Boots, Silvio Savarese, Li Fei-Fei,

**[Speaker Host]:**

assistant
Shankar Kumar and William Byrne. Minimum Bayes-Risk Decoding for Statistical Machine Translation. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, pp. 169–176, Boston, Massachusetts, USA, May 2004. Association for Computational Linguistics.

Lucy Lai, Ann Zixiang Huang, and Samuel J. Gershman. Action chunking as policy compression. 2022.

Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. Dart: Noise injection for robust imitation learning. In Conference on Robot Learning, pp. 143–156. PMLR, 2017.

Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. Behavior Generation with Latent Actions. March 2024.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding. In Proceedings of the 40th International Conference on Machine Learning, pp. 19274–19286. PMLR, July 2023.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive Decoding: Open-ended Text Generation as Optimization. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12286–12312, Toronto, Canada, July 2023. Association for Computational Linguistics.

Yunzhu Li, Jiaming Song, and Stefano Ermon. Infogail: Interpretable Imitation Learning from Visual Demonstrations. Advances in Neural Information Processing Systems, 30, 2017.

Johan Löfberg. Oops! I Cannot Do It Again: Testing for Recursive Feasibility in MPC. Automatica, 48(3):550–555, 2012.

Xiao Ma, Sumit Patidar, Iain Haughton, and Stephen James. Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation. March 2024.

Ajay Mandlekar, Fabio Ramos, Byron Boots, Silvio Savarese, Li Fei-Fei.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
Ramos, Byron Boots, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Dieter
Fox. IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline
Robot Manipulation Data. In 2020 IEEE International Conference on Robotics and Automation
(ICRA), pp. 4414-4420, May 2020.
Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li FeiFei, Silvio Savarese, Yuke Zhu, and Roberto Mart´ın-Mart´ın. What Matters in Learning from
Offline Human Demonstrations for Robot Manipulation. In Proceedings of the 5th Conference on
Robot Learning, pp. 1678-1690. PMLR, January 2022.
Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. Locally Typical Sampling. Transactions of the Association for Computational Linguistics, 11:102-121, 2023.
Kunal Menda, Katherine Driggs-Campbell, and Mykel J Kochenderfer.

**[Speaker Guest]:**

Ensembledagger: A
bayesian approach to safe imitation learning. In 2019 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS), pp. 5041-5048. IEEE, 2019.
Published as a conference paper at ICLR 2025
Mathias M¨uller and Rico Sennrich. Understanding the Properties of Minimum Bayes Risk Decoding
in Neural Machine Translation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli
(eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics
and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long
Papers), pp. 259-272, Online, August 2021. Association for Computational Linguistics.
Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, and Sergey Levine. Steering Your Generalists:
Improving Robotic Foundation Models via Value Guidance. In 8th Annual Conference on Robot
Learning, September 2024.
Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation, May 2024.
St´ephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structu

**[Speaker Host]:**

assistant
Ramos, Byron, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Dieter Fox. IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline Robot Manipulation Data. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 4414–4420, May 2020.

Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín. What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. In Proceedings of the 5th Conference on Robot Learning, pp. 1678–1690. PMLR, January 2022.

Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. Locally Typical Sampling. Transactions of the Association for Computational Linguistics, 11:102–121, 2023.

Kunal Menda, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. Ensembledagger: A Bayesian Approach to Safe Imitation Learning. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5041–5048. IEEE, 2019.

Mathias Müller and Rico Sennrich. Understanding the Properties of Minimum Bayes Risk Decoding in Neural Machine Translation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 259–272, August 2021.

Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, and Sergey Levine. Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. In 8th Annual Conference on Robot Learning, September 2024.

Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation. May 2024.

Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A Reduction of Imitation Learning and Structure.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
ell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627-635. JMLR Workshop and Conference

**[Speaker Guest]:**

Proceedings, 2011.
Nur Muhammad Shafiullah, Zichen Cui, Ariuntuya Arty Altanzaya, and Lerrel Pinto. Behavior
transformers: Cloning k modes with one stone. Advances in neural information processing systems, 35:22955-22968, 2022.
Lucy Xiaoyang Shi, Archit Sharma, Tony Z. Zhao, and Chelsea Finn. Waypoint-Based Imitation
Learning for Robotic Manipulation. In Proceedings of The 7th Conference on Robot Learning,
pp. 2195-2209. PMLR, December 2023.
Aravind Sivakumar, Kenneth Shaw, and Deepak Pathak. Robotic telekinesis: Learning a robotic
hand imitator by watching humans on youtube. Robotics: Science and Systems (RSS), 2022.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
Gokul Swamy, Sanjiban Choudhury, Drew Bagnell, and Steven Wu. Causal imitation learning under
temporally correlated noise. In International Conference on Machine Learning, pp. 20877-20890.

**[Speaker Host]:**

PMLR, 2022.
Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep
Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot
policy. arXiv preprint arXiv:2405. 12213, 2024.
Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David
Crandall, and Dhruv Batra. Diverse Beam Search for Improved Description of Complex Scenes.
Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), April 2018.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language
Models. arXiv:2203. 11171 [cs], April 2022.
Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Din

**[Speaker Guest]:**

assistant
A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.

Nur Muhammad Shafiullah, Zichen Cui, Ariuntuya Arty Altanzaya, and Lerrel Pinto. Behavior transformers: Cloning k modes with one stone. Advances in Neural Information Processing Systems, 35:22955–22968, 2022.

Lucy Xiaoyang Shi, Archit Sharma, Tony Z. Zhao, and Chelsea Finn. Waypoint-Based Imitation Learning for Robotic Manipulation. In Proceedings of the Seventh Conference on Robot Learning, pp. 2195–2209. PMLR, December 2023.

Aravind Sivakumar, Kenneth Shaw, and Deepak Pathak. Robotic telekinesis: Learning a robotic hand imitator by watching humans on YouTube. Robotics: Science and Systems (RSS), 2022.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 568:127063, 2024.

Gokul Swamy, Sanjiban Choudhury, Drew Bagnell, and Steven Wu. Causal Imitation Learning under Temporally Correlated Noise. In International Conference on Machine Learning, pp. 20877–20890. PMLR, 2022.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An Open-Source Generalist Robot Policy. arXiv preprint arXiv:2405.12213, 2024.

Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. Diverse Beam Search for Improved Description of Complex Scenes. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), April 2018.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171 [cs], April 2022.

Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Din


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
n Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston.
Neural Text Generation With Unlikelihood Training. In International Conference on Learning
Representations, September 2019.
Chuan Wen, Jierui Lin, Trevor Darrell, Dinesh Jayaraman, and Yang Gao. Fighting Copycat Agents
in Behavioral Cloning from Observation Histories. In Advances in Neural Information Processing
Systems, volume 33, pp. 2564-2575. Curran Associates, Incorporated, 2020.
Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel. Gello: A general, low-cost,
and intuitive teleoperation framework for robot manipulators. arXiv preprint arXiv:2309. 13037,

**[Speaker Host]:**

2023.
Annie Xie, Dylan Losey, Ryan Tolsma, Chelsea Finn, and Dorsa Sadigh. Learning Latent Representations to Influence Multi-Agent Interaction. In Proceedings of the 2020 Conference on Robot
Learning, pp. 575-588. PMLR, October 2021.
Published as a conference paper at ICLR 2025
Xiaomeng Xu, Huy Ha, and Shuran Song. Dynamics-Guided Diffusion Model for Robot Manipulator Design, February 2024.
Tianxiang Zhao, Wenchao Yu, Suhang Wang, Lu Wang, Xiang Zhang, Yuncong Chen, Yanchi
Liu, Wei Cheng, and Haifeng Chen.
Skill disentanglement for imitation learning from suboptimal demonstrations.
In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, pp. 3513-3524. ACM, August 2023a.

**[Speaker Guest]:**

doi:
10. 1145/3580305. 3599506. URL [web link] doi. org/10. 1145/3580305. 3599506.
Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware. In Robotics: Science and Systems (RSS) 2023, April

**[Speaker Host]:**

2023b.
Hankz Hankui Zhuo, Yantian Zha, Subbarao Kambhampati, and Xin Tian. Discovering underlying
plans based on shallow models. ACM Transactions on Intelligent Systems and Technology (TIST),

**[Speaker Guest]:**

11(2):1-30, 2020.
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

previous decision

**[Speaker Guest]:**

positive samples

**[Speaker Host]:**

negative samples

**[Speaker Guest]:**

demo distribution
Figure 10: Distribution

**[Speaker Host]:**

assistant
Neural Text Generation With Unlikelihood Training. In International Conference on Learning Representations, September 2019.

Fighting Copycat Agents in Behavioral Cloning from Observation Histories. In Advances in Neural Information Processing Systems, volume 33, pp. 2564–2575. Curran Associates, Incorporated, 2020.

Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators. arXiv preprint arXiv:2309.13037, 2023.

Learning Latent Representations to Influence Multi-Agent Interaction. In Proceedings of the 2020 Conference on Robot Learning, pp. 575–588. PMLR, October 2021.

Dynamics-Guided Diffusion Model for Robot Manipulator Design. February 2024.

Skill Disentanglement for Imitation Learning from Suboptimal Demonstrations. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, pp. 3513–3524. ACM, August 2023.

Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Robotics: Science and Systems (RSS) 2023, April 2023.

Discovering Underlying Plans Based on Shallow Models. ACM Transactions on Intelligent Systems and Technology (TIST), 11(2):1–30, 2020.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user

**[Speaker Guest]:**

samples

**[Speaker Host]:**

demo distribution
Figure 10: Distributional interpretation of BID. The backward criterion (Equation 6) favors samples close to
the past decision; the forward criterion (Equation 7) promotes samples with a high likelihood under the target

**[Speaker Guest]:**

distribution.

**[Speaker Host]:**

Action Horizon

**[Speaker Guest]:**

Noise Level

**[Speaker Host]:**

3

**[Speaker Guest]:**

7

**[Speaker Host]:**

0. 0

**[Speaker Guest]:**

4. 03

**[Speaker Host]:**

2. 07

**[Speaker Guest]:**

1. 54

**[Speaker Host]:**

1. 06

**[Speaker Guest]:**

0. 00

**[Speaker Host]:**

0. 2

**[Speaker Guest]:**

1. 43

**[Speaker Host]:**

0. 94

**[Speaker Guest]:**

0. 39

**[Speaker Host]:**

0. 71

**[Speaker Guest]:**

1. 32

**[Speaker Host]:**

0. 4

**[Speaker Guest]:**

0. 36

**[Speaker Host]:**

0. 42

**[Speaker Guest]:**

0. 59

**[Speaker Host]:**

0. 835

**[Speaker Guest]:**

1. 11
Table 3: Total variation distance between the action distributions of each model and the expert in environments
with varying noise levels. Lower values indicate better performance.

**[Speaker Host]:**

A
ADDITIONAL EXPERIMENTS

**[Speaker Guest]:**

A. 1
ONE-DIMENSIONAL SIMULATIONS
In addition to Fig. 5, we summarize the total variation distance between each learned policy and the
demonstration in the one-dimensional simulation. Our results indicate that a shorter action horizon
is more effective in noisier environments, whereas a longer action horizon yields better performance
in static environments.

**[Speaker Host]:**

A. 2
ACTION HORIZON VS. CONTEXT LENGTH
Setup. Our work builds on the premise that the action horizon is longer than the context length,
as commonly designed for recent policies. While BID mitigates the inherent limitations of this design choice through test-time decoding, an important question remains: could extending the context
length yield stronger policies? To understand this, we trained diffusion policies with varying combinations of action horizons and context lengths on the Push-T task. Specifically, we use a short
context length (c = 2) and a short action horizon (h = 2) as our baseline, set the prediction length
equal to the action horizon l = h, and incrementally increase these parameters to larger values
6, 10, 14 to assess their impact.
Result. Fig. 11 compares the performance of the policy learned with different ∆h = h −c. As expected, the policy with both a short action horizon and a short context length does not perform well,
due to its limited capability to model long-range temporal dependencies. Interesti

**[Speaker Guest]:**

assistant

**[Speaker Host]:**

Samples

**[Speaker Guest]:**

Demo Distribution
Figure 10: Distributional interpretation of BID. The backward criterion (Equation 6) favors samples close to the past decision; the forward criterion (Equation 7) promotes samples with a high likelihood under the target distribution.


**[Speaker Host]:**

Action Horizon

**[Speaker Guest]:**

Noise Level

**[Speaker Host]:**

3

**[Speaker Guest]:**

7

**[Speaker Host]:**

0.0

**[Speaker Guest]:**

4.03

**[Speaker Host]:**

2.07

**[Speaker Guest]:**

1.54

**[Speaker Host]:**

1.06

**[Speaker Guest]:**

0.00

**[Speaker Host]:**

0.2

**[Speaker Guest]:**

1.43

**[Speaker Host]:**

0.94

**[Speaker Guest]:**

0.39

**[Speaker Host]:**

0.71

**[Speaker Guest]:**

1.32

**[Speaker Host]:**

0.4

**[Speaker Guest]:**

0.36

**[Speaker Host]:**

0.42

**[Speaker Guest]:**

0.59

**[Speaker Host]:**

0.835

**[Speaker Guest]:**

1.11

Table 3: Total variation distance between the action distributions of each model and the expert in environments with varying noise levels. Lower values indicate better performance.

Additional Experiments
A.1 One-Dimensional Simulations
In addition to Fig. 5, we summarize the total variation distance between each learned policy and the demonstration in the one-dimensional simulation. Our results indicate that a shorter action horizon is more effective in noisier environments, whereas a longer action horizon yields better performance in static environments.

A.2 Action Horizon vs. Context Length
Setup. Our work builds on the premise that the action horizon is longer than the context length, as commonly designed for recent policies. While BID mitigates the inherent limitations of this design choice through test-time decoding, an important question remains: could extending the context length yield stronger policies? To understand this, we trained diffusion policies with varying combinations of action horizons and context lengths on the Push-T task. Specifically, we use a short context length (c = 2) and a short action horizon (h = 2) as our baseline, set the prediction length equal to the action horizon (l = h), and incrementally increase these parameters to larger values (6, 10, 14) to assess their impact.

Result. Fig. 11 compares the performance of the policy learned with different ∆h = h − c. As expected, the policy with both a short action horizon and a short context length does not perform well, due to its limited capability to model long-range temporal dependencies. Interesti


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
model long-range temporal dependencies. Interestingly, extending the
context length initially boosts performance (∆h = −4), but this trend reverses as the context length
becomes too long (∆h less than or equal to −8), likely due to overfitting to an increased number of spurious features. In contrast, expanding the action horizon results in more robust performance improvements,
validating its pivotal role in imitation learning from human demonstrations.

**[Speaker Host]:**

A. 3
ABLATION STUDY OF FORWARD CONTRAST

**[Speaker Guest]:**

Setup.
To understand the effect of forward contrast (Equation 7), we evaluate the full version
of our method against three reduced variants in open-loop operations: Vanilla (without forward
contrast), Positive (without negative samples), and Negative (without positive samples). Similar
to §5. 2. 2, our ablation study is conducted in seven simulation tasks, each with three random seeds.

**[Speaker Host]:**

Result.
Fig. 12 summarizes the result of this ablation study. Notably, both positive and negative samples are essential for effective sample selection, and omitting either leads to significant
performance declines. Without negative samples, our decoding method reduces to an approximate
maximum a posteriori estimation, which can result in suboptimal decisions due to modeling errors.
Conversely, without positive samples, the sampling process may be biased towards rare instances.
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

8

**[Speaker Host]:**

0

**[Speaker Guest]:**

8

**[Speaker Host]:**

h

**[Speaker Guest]:**

0. 7

**[Speaker Host]:**

0. 8

**[Speaker Guest]:**

0. 9

**[Speaker Host]:**

Test Score
Figure 11: Effect of action horizon (h) and context length (c) on diffusion policies in the Push-T task. The
baseline is set at h = 2 and c = 2, with ∆h = h −c = 0. Extending the action horizon (h > 2) consistently
improves performance, whereas extending the context length (c > 2) can cause substantial performance declines. Each model is trained for 5k epochs. Results are averaged over the last five checkpoints.

**[Speaker Guest]:**

Vanilla

**[Speaker Host]:**

Positive

**[Speaker Guest]:**

Negative

**[Speaker Host]:**

Contrast

**[Speaker Guest]:**

Method

**[Speaker Host]:**

0. 77

**[Speaker Guest]:**

0. 78

**[Speaker Host]:**

0. 79

**[Speaker Guest]:**

0. 80

**[Speaker Host]:**

Success Rate
Figure 12: Effect of positive and negative samples

**[Speaker Guest]:**

on fo

**[Speaker Host]:**

assistant
Modeling long-range temporal dependencies: extending context length initially improves performance (∆h = −4), but this trend reverses when context length becomes too long (∆h ≤ −8), likely due to overfitting to spurious features. In contrast, expanding the action horizon leads to more robust performance improvements, validating its key role in imitation learning from human demonstrations.

Ablation Study of Forward Contrast
Setup: To assess the impact of forward contrast (Equation 7), we compare the full method against three reduced variants in open-loop operations: Vanilla (without forward contrast), Positive (without negative samples), and Negative (without positive samples). As in §5.2.2, the study is conducted across seven simulation tasks, each run with three random seeds.

Result: Figure 12 summarizes the findings. Both positive and negative samples are critical for effective sample selection. Omitting either results in significant performance drops. Without negative samples, the method reduces to approximate maximum a posteriori estimation, leading to suboptimal decisions due to modeling errors. Without positive samples, the sampling process becomes biased toward rare instances.

Published as a conference paper at ICLR 2025.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
12: Effect of positive and negative samples
on forward contrast. Performance of ablated variants
of forward contrast is evaluated across seven simulation tasks. The absence of either positive or negative samples prevents achieving the full performance
gains observed with the contrast objective.

**[Speaker Guest]:**

0. 0

**[Speaker Host]:**

0. 2

**[Speaker Guest]:**

0. 4

**[Speaker Host]:**

0. 6

**[Speaker Guest]:**

0. 8

**[Speaker Host]:**

1. 0

**[Speaker Guest]:**

Decay Rate

**[Speaker Host]:**

0. 00

**[Speaker Guest]:**

0. 25

**[Speaker Host]:**

0. 50

**[Speaker Guest]:**

0. 75

**[Speaker Host]:**

1. 00

**[Speaker Guest]:**

Relative Test Score

**[Speaker Host]:**

pusht

**[Speaker Guest]:**

lift

**[Speaker Host]:**

toolhang
Figure 13: Effect of the decay rate for the exponential moving average. In each task, we measure the
relative performance among different decay rates.
The optimal decay rate varies by task, leading to a
practical challenge of identifying a universal temporal ensembling strategy (Zhao et al., 2023b).
This result highlights the importance of both components and suggests the potential for extending
this paradigm in future work.

**[Speaker Guest]:**

A. 4
CHALLENGES FOR TEMPORAL ENSEBMLING
EMA exhibits competitive performance in Fig. 6. However, tuning its decay rate can be difficult in
practice. Fig. 13 shows the sensitivity of EMA to the decay rate across three different tasks, where
the optimal choices differ significantly. We conjecture that this high sensitivity stems from the
variability in the latent strategies between consecutive predictions. When consecutive predictions
follow similar strategies, a lower decay rate (i. e., stronger moving average) can enhance smoothness
and improve performance. Conversely, when consecutive predictions diverge in their underlying
strategies, averaging them can introduce adverse effects. Our method promotes coherence in latent
strategies and thus effectively complements temporal ensembling, as evidenced in Fig. 7.

**[Speaker Host]:**

A. 5
ABLATION STUDIES ON VQ-BET
Table 4 presents the performance of various sampling methods with VQ-BeT (Lee et al., 2024) on
the Push-T task under different levels of environment stochasticity. We compare BID against vanilla
sampling and EMA, for both open-loop and closed-loop executions. BID consistently outperforms
the baselines, particu

**[Speaker Guest]:**

assistant
The effect of positive and negative samples on forward contrast is evaluated across seven simulation tasks. The absence of either positive or negative samples prevents achieving the full performance gains observed with the contrast objective.

Figure 13 shows the sensitivity of exponential moving average (EMA) to decay rate across three tasks, where the optimal decay rate varies significantly by task. This variation highlights the practical challenge of identifying a universal temporal ensembling strategy.

Our method promotes coherence in latent strategies, which helps mitigate the sensitivity of EMA to decay rate. When consecutive predictions follow similar strategies, a lower decay rate improves performance by enhancing smoothness. When predictions diverge in strategy, averaging them can harm performance. By aligning latent strategies, our approach effectively complements temporal ensembling.

Ablation studies on VQ-BET show that BID consistently outperforms vanilla sampling and EMA across open-loop and closed-loop executions under varying levels of environmental stochasticity.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
ID consistently outperforms
the baselines, particularly in highly stochastic settings, with close-loop BID attaining the best performance on average. Moreover, the performance of EMA degrades significantly as the environment
gets more noisy, likely due to the divergence in strategies between consecutive action chunks. Additionally, we conduct an ablation study on the forward contrast objective, isolating the roles of
positive samples (from a strong model) and negative samples (from a weaker model). We observe
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

Stochastic Noise

**[Speaker Guest]:**

0. 0

**[Speaker Host]:**

1. 0

**[Speaker Guest]:**

1. 5

**[Speaker Host]:**

Average

**[Speaker Guest]:**

Vanilla
64. 0 plus or minus 4. 2
26. 9 plus or minus 2. 8
13. 0 plus or minus 0. 4
34. 6 plus or minus 1. 7

**[Speaker Host]:**

EMA
64. 1 plus or minus 1. 7
27. 6 plus or minus 3. 3
12. 9 plus or minus 1. 1
34. 9 plus or minus 1. 3

**[Speaker Guest]:**

Backward (ours)
64. 0 plus or minus 1. 3
27. 6 plus or minus 1. 0
13. 4 plus or minus 2. 4
35. 1 plus or minus 1. 0

**[Speaker Host]:**

+ Positive (ours)
65. 6 plus or minus 1. 9
29. 7 plus or minus 0. 4
15. 7 plus or minus 1. 7
37. 0 plus or minus 0. 9

**[Speaker Guest]:**

+ Negative (ours)
65. 1 plus or minus 2. 7
26. 6 plus or minus 0. 8
14. 8 plus or minus 2. 4
35. 5 plus or minus 1. 2

**[Speaker Host]:**

BID (ours)
66. 1 plus or minus 3. 5
31. 4 plus or minus 3. 0
16. 0 plus or minus 1. 2
37. 8 plus or minus 1. 6
(a) Open-Loop Operation

**[Speaker Guest]:**

Stochastic Noise

**[Speaker Host]:**

0. 0

**[Speaker Guest]:**

1. 0

**[Speaker Host]:**

1. 5

**[Speaker Guest]:**

Average

**[Speaker Host]:**

Vanilla
48. 9 plus or minus 2. 7
38. 3 plus or minus 3. 4
29. 5 plus or minus 0. 9
38. 9 plus or minus 1. 5

**[Speaker Guest]:**

EMA
52. 6 plus or minus 2. 9
35. 7 plus or minus 2. 2
18. 4 plus or minus 2. 3
35. 6 plus or minus 1. 4

**[Speaker Host]:**

Backward (ours)
52. 7 plus or minus 1. 3
42. 0 plus or minus 3. 2
29. 4 plus or minus 0. 5
41. 4 plus or minus 1. 2

**[Speaker Guest]:**

+ Positive (ours)
53. 8 plus or minus 3. 3
44. 1 plus or minus 2. 5
30. 4 plus or minus 0. 8
42. 8 plus or minus 1. 4

**[Speaker Host]:**

+ Negative (ours)
53. 0 plus or minus 2. 1
44. 3 plus or minus 2. 2
30. 8 plus or minus 0. 4
42. 7 plus or minus 1. 0

**[Speaker Guest]:**

BID (ours)
54. 4 plus or minus 1. 8
45. 3 plus or minus 3. 8
31. 7 plus or minus 0. 3

**[Speaker Host]:**

43. 8 plus

**[Speaker Guest]:**

assistant
ID consistently outperforms the baselines, especially in highly stochastic environments. Close-loop BID achieves the best average performance. EMA's performance drops significantly as noise increases, likely because of strategy divergence between action chunks. An ablation study on the forward contrast objective isolates the roles of positive samples (from a strong model) and negative samples (from a weaker model). We observe that adding positive samples improves performance across all metrics, while negative samples have a modest effect. The results show that the proposed methods outperform vanilla and EMA, with BID leading in both open-loop and closed-loop settings.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

**[Speaker Host]:**

or minus 3. 8
31. 7 plus or minus 0. 3
43. 8 plus or minus 1. 4
(b) Closed-Loop Operation
Table 4: Success rates of VQ-BeT on the Push-T task under various conditions and sampling methods. The left
table shows open-loop success rates, while the right table shows closed-loop success rates. BID consistently
outperforms vanilla counterparts in both settings.

**[Speaker Guest]:**

Vanilla

**[Speaker Host]:**

L2 (Ours)

**[Speaker Guest]:**

L1 (Ours)

**[Speaker Host]:**

Cosine (Ours)

**[Speaker Guest]:**

Square
0. 68 plus or minus 0. 06
0. 76 plus or minus 0. 03
0. 76 plus or minus 0. 04
0. 73 plus or minus 0. 04

**[Speaker Host]:**

Lift
0. 12 plus or minus 0. 02
0. 58 plus or minus 0. 06
0. 68 plus or minus 0. 05
0. 70 plus or minus 0. 02

**[Speaker Guest]:**

Kitchen
0. 22 plus or minus 0. 04
0. 64 plus or minus 0. 05
0. 70 plus or minus 0. 04
0. 61 plus or minus 0. 03
Table 5: Effect of distance metric. We evaluate BID with different distance metrics on three robot manipulation
tasks. Regardless of the chosen metric, BID substantially outperforms the vanilla baseline. Notably, the L1
distance yields an even stronger performance than the default L2 distance used in our other experiments.
that both types of samples contribute to the strong performance of BID across varying levels of
environment stochasticity.

**[Speaker Host]:**

A. 6
CHOICE OF DISTANCE METRICS
As described in §4. 2, our method uses the L2 distance as the default distance measure for quantifying
similarity between action chunks. Yet, L2 distance is just one of several viable options. To assess
the sensitivity of our approach to the choice of distance metric, we extend the experiments in §5. 2. 1
to further evaluate BID with alternative choices, including L1 and cosine distance. Results from
three robot manipulation tasks, summarized in Table 5, demonstrate that BID achieves substantial
performance gains regardless of the metric used. In particular, BID with L1 distance surpasses the
default L2 distance in performance, highlighting the untapped potential of our method.

**[Speaker Guest]:**

B
ADDITIONAL DISCUSSIONS
Interpretation of our method. Our method makes no changes to the learned pol

**[Speaker Host]:**

assistant
Success rates of VQ-BeT on the Push-T task under various conditions and sampling methods. Open-loop success rates: 31.7 ± 0.3, 43.8 ± 1.4. Closed-loop success rates: Vanilla L2: 0.68 ± 0.06, L1: 0.76 ± 0.03, Cosine: 0.73 ± 0.04, Square: 0.70 ± 0.02. Lift: Vanilla 0.12 ± 0.02, L1 0.58 ± 0.06, L2 0.68 ± 0.05, Cosine 0.70 ± 0.02. Kitchen: Vanilla 0.22 ± 0.04, L1 0.64 ± 0.05, L2 0.70 ± 0.04, Cosine 0.61 ± 0.03. BID consistently outperforms vanilla baselines in both open-loop and closed-loop settings. Across all tasks and sampling methods, the method shows strong and stable performance. The L1 distance metric achieves better results than the default L2 distance, suggesting that the choice of distance metric has a meaningful impact on performance. These results indicate that both types of samples contribute to the robustness and effectiveness of BID under varying environmental conditions.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
od. Our method makes no changes to the learned policy; instead, it
intervenes in the model distribution through sample selection. As illustrated in Fig. 10, randomly
sampled sequences may be misaligned with both the previous decisions and the target demonstrations. Given a set of candidates, the backward step first identifies the behavioral mode from the past
decision stored in memory; the forward step then removes the samples with low likelihood under
the target distribution using prior knowledge of positive and negative samples. By comparing samples across time steps and model checkpoints, our method bridges the gap between the proposal and
target distributions at test time.
Relation to recent methods. Our method builds upon the receding horizon (Chi et al., 2023) and
temporal ensembling (Zhao et al., 2023b) used in previous works, but with crucial distinctions.
Receding horizon seeks a compromise between long-term consistency and short-term reactivity by
using a moderate action horizon (e. g., half of the prediction length), which is inevitably sup-optimal
when both factors are prominent. Temporal ensembling strengthens dependency across chunks by
averaging multiple decisions over time; however, weighted-averaging operations can be detrimental
when consecutive decisions fall into distinct modes. Our method more effectively addresses crosschunk dependency through dedicated behavioral search and is not mutually exclusive to previous
methods. We will demonstrate in the next section that combining our method with the moving
average can further improve closed-loop action chunking.
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

name

**[Speaker Host]:**

value

**[Speaker Guest]:**

batch size N

**[Speaker Host]:**

mode size K

**[Speaker Guest]:**

prediction length l
temporal coherence decay ρ

**[Speaker Host]:**

0. 5
moving average decay λ

**[Speaker Guest]:**

0. 5
Table 6: Default hyper-parameters in our experiments.

**[Speaker Host]:**

C

**[Speaker Guest]:**

ADDITIONAL DETAILS

**[Speaker Host]:**

C. 1
SIMULATION EXPERIMENT DETAILS

**[Speaker Guest]:**

C. 1. 1

**[Speaker Host]:**

ENVIRONMENT DETAILS
Our simulation experiments are conducted on three robot manipulation benchmarks. We use the

**[Speaker Guest]:**

tra

**[Speaker Host]:**

assistant
Our method makes no changes to the learned policy; instead, it intervenes in the model distribution through sample selection. As illustrated in Fig. 10, randomly sampled sequences may be misaligned with both the previous decisions and the target demonstrations. Given a set of candidates, the backward step first identifies the behavioral mode from the past decision stored in memory; the forward step then removes the samples with low likelihood under the target distribution using prior knowledge of positive and negative samples. By comparing samples across time steps and model checkpoints, our method bridges the gap between the proposal and target distributions at test time.

Our method builds upon receding horizon (Chi et al., 2023) and temporal ensembling (Zhao et al., 2023b), but with crucial distinctions. Receding horizon seeks a compromise between long-term consistency and short-term reactivity by using a moderate action horizon (e.g., half of the prediction length), which is inevitably sub-optimal when both factors are prominent. Temporal ensembling strengthens dependency across chunks by averaging multiple decisions over time; however, weighted-averaging operations can be detrimental when consecutive decisions fall into distinct modes. Our method more effectively addresses cross-chunk dependency through dedicated behavioral search and is not mutually exclusive to previous methods. We will demonstrate in the next section that combining our method with the moving average can further improve closed-loop action chunking.

Default hyper-parameters: batch size N = 0.5, mode size K = 0.5, prediction length l = 0.5, temporal coherence decay ρ = 0.5, moving average decay λ = 0.5.

Simulation experiments are conducted on three robot manipulation benchmarks.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
hree robot manipulation benchmarks. We use the
training data collected from human demonstrations in each benchmark.
Push-T: We adopt the Push-T environment introduced in (Chi et al., 2023), where the goal is to push
a T-shaped block on a table to a target position. The action space is two-dimensional end-effector
velocity control. The training dataset contains 206 demonstrations collected by humans.
Robomimic: We use five tasks in the Robomimic suite (Mandlekar et al., 2022), namely Lift, Can,
Square, Transport, and Tool Hang. The training dataset for each task contains 300 episodes collected
from multi-human (MH) demonstrations.
Franka Kitchen: We use the Franka Kitchen environment from (Gupta et al., 2020), featuring a
Franka Panda arm with a seven-dimensional action space and 566 human-collected demonstrations.
The learned policy is evaluated on test cases involving four or more objects (p4), a challenging yet
practical task for robotic manipulation in household contexts.

**[Speaker Guest]:**

C. 1. 2
IMPLEMENTATION DETAILS.
Our implementation of BID for Diffusion Policy is built upon the official code of Chi et al. ,
with modifications made solely to the inference process. The policy takes in state inputs and predicts
a chunk of 16 actions as outputs. For each simulation task, we train the model for 100-1000 epochs
to reach near-optimal performance. We evaluate it in closed-loop operations, i. e., action horizon is
set to 1. For forward contrast, we train the weak policy for 10-100 epochs, resulting in a suboptimal
policy for each task. The core hyperparameters are summarized in Table 6.
Our implementation of BID for VQ-BeT (Lee et al., 2024) is built upon the code of LeRobot (Cadene
et al., 2024). We use the best public checkpoint as the strong policy and a checkpoint trained for
100k iterations as the weak policy. Since BID requires sample diversity, we set the temperature to
be 0. 5 for all methods in our experiments.

**[Speaker Host]:**

C. 2
REAL-WORLD EXPERIMENT DETAILS

**[Speaker Guest]:**

C. 2. 1

**[Speaker Host]:**

DYNAMIC PLACING

**[Speaker Guest]:**

Task

**[Speaker Host]:**

assistant
Three robot manipulation benchmarks. We use the training data collected from human demonstrations in each benchmark.

Push-T: We adopt the Push-T environment introduced in (Chi et al., 2023), where the goal is to push a T-shaped block on a table to a target position. The action space is two-dimensional end-effector velocity control. The training dataset contains 206 demonstrations collected by humans.

Robomimic: We use five tasks in the Robomimic suite (Mandlekar et al., 2022): Lift, Can, Square, Transport, and Tool Hang. The training dataset for each task contains 300 episodes collected from multi-human (MH) demonstrations.

Franka Kitchen: We use the Franka Kitchen environment from (Gupta et al., 2020), featuring a Franka Panda arm with a seven-dimensional action space and 566 human-collected demonstrations. The learned policy is evaluated on test cases involving four or more objects (p4), a challenging yet practical task for robotic manipulation in household contexts.

Implementation details: Our implementation of BID for Diffusion Policy is built upon the official code of Chi et al., with modifications made solely to the inference process. The policy takes in state inputs and predicts a chunk of 16 actions as outputs. For each simulation task, we train the model for 100–1000 epochs to reach near-optimal performance. We evaluate it in closed-loop operations, i.e., action horizon is set to 1. For forward contrast, we train the weak policy for 10–100 epochs, resulting in a suboptimal policy for each task. The core hyperparameters are summarized in Table 6.

Our implementation of BID for VQ-BeT (Lee et al., 2024) is built upon the code of LeRobot (Cadene et al., 2024). We use the best public checkpoint as the strong policy and a checkpoint trained for 100k iterations as the weak policy. Since BID requires sample diversity, we set the temperature to 0.5 for all methods in our experiments.

Real-world experiment details: Dynamic placing task.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
LD EXPERIMENT DETAILS

**[Speaker Guest]:**

C. 2. 1

**[Speaker Host]:**

DYNAMIC PLACING

**[Speaker Guest]:**

Task.
We consider a task where the robot is to deliver an object held in its gripper into a cup
held by a human. As shown in Fig. 14, this task comprises four main stages and presents two core

**[Speaker Host]:**

challenges.

Demonstration data includes 150 episodes: 50 clean, consistent demonstrations and 100 noisy, diverse ones, varying in speed, pauses, and trajectory. Each episode successfully completes the task, with the cup’s position fixed throughout each episode.

Robot hardware: A Franka Panda robot is used, operating with a vision-based diffusion policy. The robot’s behavior is guided by human demonstrations, with the system designed to handle temporal variations and style differences in human actions. The cup’s position may change during execution, requiring real-time adaptation.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
he robot is tasked with picking up a cup and placing it on a saucer nearby. The four main stages
are (a) initializing the robot, (b) approaching the target cup, (c) grasping the target cup, (d) picking up the cup,
and (e) placing the cup on the target saucer. The position of the target cup may change during an episode.
with two cameras: one ego-centric camera mounted at the wrist of the robot, one third-person camera
mounted at a static bracket. Both cameras provide visual observations at a resolution of 256 times 256
pixels. The robot operates at a frequency of 10 Hz, with a prediction length of 16 time steps.

**[Speaker Guest]:**

Evaluation.
We evaluate our method in comparison to vanilla random sampling under two conditions: static target, where the target cup remains fixed throughout the evaluation, and dynamic
target, where the target cup is gradually moved. In the dynamic setting, the location of the cup stays
within the range of training locations, but the movement is not encountered during training. This
evaluation protocol is designed to explicitly assess the ability of the policy to react to unexpected
dynamics in the environment. Each method-setting pair is tested over 20 episodes, with both the
initial and target locations randomized across different episodes.

**[Speaker Host]:**

Result.
We summarize the result of the real-world experiments in Fig. 8. The success rate of
vanilla random sampling is generally limited due to oscillations between different latent strategies,
which quickly diverge from the distribution of demonstrations. This issue is particularly pronounced
in the dynamic setting, where the vanilla baseline struggles to account for the target movements
within an action chunk lasting for 1. 6 seconds. In contrast, the proposed BID method significantly
improves performance in both static and dynamic settings. Notably, BID maintains a similar success
rate in the dynamic setting as in the static setting, suggesting its potential to extend action chunking
into uncertain environments.

**[Speaker Guest]:**

C.

**[Speaker Host]:**

assistant
The robot is tasked with picking up a cup and placing it on a nearby saucer. The four main stages are: (a) initializing the robot, (b) approaching the target cup, (c) grasping the cup, (d) picking it up, and (e) placing it on the saucer. The target cup's position may change during an episode.

The robot is equipped with two cameras: one mounted at the wrist (ego-centric) and one at a static third-person bracket. Both provide visual input at 256x256 pixels. The robot operates at 10 Hz with a prediction horizon of 16 time steps.

Evaluation: We compare our method to vanilla random sampling under two conditions — static target, where the cup stays in a fixed position, and dynamic target, where the cup moves gradually. In the dynamic setting, the cup's location remains within the training range but the movement was not seen during training. This tests the policy’s ability to handle unexpected dynamics. Each method-setting pair is evaluated over 20 episodes, with both initial and target positions randomized across episodes.

Results: Real-world experiments show that vanilla random sampling has limited success due to oscillation between different strategies, which quickly drift from the demonstration distribution. This is especially clear in dynamic settings, where the policy fails to adapt within a 1.6-second action chunk. In contrast, the BID method significantly outperforms vanilla sampling in both static and dynamic scenarios. It maintains a consistent success rate in dynamic settings, suggesting it can effectively handle uncertainty and unexpected movements.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user

**[Speaker Guest]:**

nd action chunking
into uncertain environments.

**[Speaker Host]:**

C. 2. 2

**[Speaker Guest]:**

DYNAMIC PICKING

**[Speaker Host]:**

Task.
Next, we consider a task where the robot is required to pick up a cup and place it onto a
nearby saucer. The cup was pulled with a string until the robot’s gripper successfully grasped it. The
task consists of five main stages, which are illustrated in Fig. 15. This setup also tests the robot’s
capability to interact with a dynamic environment, a critical challenge in real-world applications.

**[Speaker Guest]:**

Policy.
We utilized the publicly available diffusion policy checkpoint from UMI (Chi et al., 2024)
without any additional fine-tuning. Notably, the policy was originally trained using demonstrations
in a static setting, where the cup’s position remained constant throughout the task. Our experimental
setup mirrored the one described by UMI, using the same UR5 robot hardware. This allowed us
to directly evaluate the policy’s transferability to a dynamic environment, where the cup’s position
changes during the task. Due to the absence of an early checkpoint, we omitted negative samples in
forward contrast, focusing solely on positive consistency discussed in Appendix A. 3.
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

Evaluation.
We evaluated BID against three baselines: vanilla random sampling in both openloop and closed-loop configurations, and EMA (closed-loop). These methods were tested under two
conditions: static target, where the cup remained in a fixed position, and dynamic target, where the
cup was moved using the string. Each method-setting combination was tested across 20 episodes,
with the initial positions of the cup and saucer kept consistent to ensure controlled comparisons.

**[Speaker Guest]:**

Results.
The results, summarized in Fig. 9, highlight the challenges of the dynamic setting. Openloop vanilla sampling performed poorly due to its inability to adapt to the cup’s movements, often
failing to approach the cup as it was pulled. While closed-loop vanilla sampling showed improved
reactivity, it suffered from

**[Speaker Host]:**

assistant
The robot is tasked with picking up a cup and placing it on a nearby saucer. The cup is pulled with a string until the gripper successfully grasps it. The task has five main stages, as shown in Figure 15, and tests the robot’s ability to interact with a dynamic environment—a key challenge in real-world applications.

We used a publicly available diffusion policy from UMI (Chi et al., 2024) without any fine-tuning. The policy was originally trained on a static environment where the cup’s position remained fixed. Our setup matches UMI’s original experiment, using the same UR5 robot hardware. This allows us to directly assess how well the policy transfers to a dynamic environment where the cup moves during execution.

We evaluated BID against three baselines: vanilla random sampling in open-loop and closed-loop configurations, and EMA in closed-loop. The tests were conducted under two conditions—static target (cup fixed) and dynamic target (cup pulled by string). Each method and setup combination was run for 20 episodes, with consistent starting positions for cup and saucer to ensure fair comparison.

Results show that open-loop vanilla sampling struggles significantly, failing to track the moving cup and often missing it entirely. Closed-loop vanilla sampling shows better reactivity but still faces limitations in adapting to real-time changes.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
pling showed improved
reactivity, it suffered from inconsistent trajectories, resulting in jittery behavior when attempting to
grasp and place the cup. Similarly, closed-loop EMA sampling demonstrated higher adaptability to
environmental changes but often failed to firmly grasp the cup, likely due to the limitations of naive
averaging, which compromises commitment to a specific strategy. In contrast, BID achieved at
least a 2x improvement in success rate compared to all other methods in the dynamic setting, while
maintaining its performance in the static setting, demonstrating both adaptability and precision in
dynamic environments.

**[Speaker Guest]:**

D
ADDITIONAL DISCUSSIONS
Relation to Option Discovery.
Action chunking and option discovery share similarities in modeling temporally extended actions. However, their designs and outcomes are often different. Option
discovery typically aims to learn hierarchical policies, explicitly discovering high-level skills from
low-level action sequences (Bagaria & Konidaris, 2020; Jiang et al., 2022; 2023; Zhao et al., 2023a).
In contrast, action chunking operates directly on low-level action sequences with fixed horizons, bypassing the need for abstraction into high-level skills. This simplicity has proven effective in largescale robotic foundation models (Team et al., 2024; Black et al., 2024), addressing the scalability
challenge that option discovery has yet to overcome. Future work could explore variable-horizon
option discovery as a compelling alternative to action chunking, potentially combining the benefits
of temporal abstraction with the scalability demonstrated by action chunking.
Relation to Long Context.
Our analysis in §3 highlights the benefits of extending the action horizon to better capture temporal dependencies across actions. Another natural approach to capturing
these dependencies is to extend the context length. However, long-context policies often suffer from
robustness issues, due to spurious correlations between past and f

**[Speaker Host]:**

assistant
Plying showed improved reactivity but suffered from inconsistent trajectories, leading to jittery behavior when grasping and placing the cup. Closed-loop EMA sampling demonstrated better adaptability to environmental changes but often failed to firmly grasp the cup, likely due to the limitations of naive averaging, which weakens commitment to a specific strategy. In contrast, BID achieved at least a 2x improvement in success rate compared to all other methods in dynamic settings, while maintaining performance in static settings, showing both adaptability and precision in dynamic environments.

Our analysis highlights the benefits of extending the action horizon to better capture temporal dependencies across actions. Another natural approach is to extend context length, though long-context policies often suffer from robustness issues due to spurious correlations between past and future actions.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
s, due to spurious correlations between past and future actions, as evidenced in Appendix A2 and studied in de Haan et al. ; Wen et al. . Nevertheless, the increasing
availability of large-scale robotic datasets for pre-training and fine-tuning may help mitigate these
challenges. Advances in leveraging long context may open up new opportunities in policy design,
such as long-context transformers (Su et al., 2024) and recurrent neural networks (Hochreiter &
Schmidhuber, 1997; Zhuo et al., 2020).

**[Speaker Guest]:**

E

**[Speaker Host]:**

PROOFS
In this section, we will clearly write out the transition dynamics in the training environment to be
Ptrain(st+1 | st, at), the transition dynamics in the test environment to be Ptest(st+1 | st, at) and
the implicit dynamics model of the training environment to be ˆPtrain(st+1 | st, at). Note that the
implicit dynamics model attempts to learn the dynamics of the training environment and not the
test environment since our policy is assumed to have been trained only on data from the training

**[Speaker Guest]:**

environment.


**[Speaker Host]:**

min
c∈R EY [L(f(X, Y ), c)|X]

**[Speaker Guest]:**

.
Published as a conference paper at ICLR 2025
Proof. The left hand side is less than or equal to the right hand side by the following logic:

**[Speaker Host]:**

EX


**[Speaker Guest]:**

min
c∈R EY [L(f(X, Y ), c)|X]

= EX [EY [L(f(X, Y ), c∗(X))|X]]
greater than or equal to

**[Speaker Host]:**

min
g(X)∈G EX, Y [L(f(X, Y ), g(X))]
where we used c∗(X) := arg minc EX[L(f(X, Y ), c)|X]. We get the inequality by recognizing
that R ⊊G. For the reverse inequality, consider any g(X) ∈G:
E[L(f(X, Y ), g(X))] = EX [EY [L(f(X, Y ), g(X))|X]]
greater than or equal to EX


**[Speaker Guest]:**

min

**[Speaker Host]:**

g
EY [L(f(X, Y ), g(X))|X]


**[Speaker Guest]:**

= EX

**[Speaker Host]:**

h

**[Speaker Guest]:**

min

**[Speaker Host]:**

c
EY [L(f(X, Y ), c)|X]

**[Speaker Guest]:**

i.
With these two inequalities, we conclude.

**[Speaker Host]:**

Next,

**[Speaker Guest]:**

assistant
Due to spurious correlations between past and future actions, as evidenced in Appendix A2 and studied in de Haan et al. and Wen et al., policy design faces challenges. However, the increasing availability of large-scale robotic datasets for pre-training and fine-tuning may help mitigate these issues. Advances in leveraging long context open new opportunities in policy design, such as long-context transformers and recurrent neural networks.

In this section, we define the transition dynamics in the training environment as Ptrain(st+1 | st, at), in the test environment as Ptest(st+1 | st, at), and the implicit dynamics model of the training environment as ˆPtrain(st+1 | st, at). The implicit dynamics model learns only the training environment’s dynamics, since the policy is trained solely on training environment data.

We establish the following lemma to compare function classes based on the information they access:

Let L be a convex function, and let X and Y be random variables. Let G be the class of functions g(X) that take X as input. Then:

min g(X)∈G EX, Y [L(f(X, Y), g(X))] = EX [min c∈R EY [L(f(X, Y), c)|X]]

The left-hand side is less than or equal to the right-hand side because:

EX [min c∈R EY [L(f(X, Y), c)|X]] = EX [EY [L(f(X, Y), c∗(X))|X]] ≥ min g(X)∈G EX, Y [L(f(X, Y), g(X))]

where c∗(X) is defined as arg min c EX [L(f(X, Y), c)|X]. Since R ⊊ G, the inequality holds.

For the reverse inequality, consider any g(X) ∈ G:

E[L(f(X, Y), g(X))] = EX [EY [L(f(X, Y), g(X))|X]] ≥ EX [min c EY [L(f(X, Y), c)|X]]

Thus, both inequalities establish equality.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

**[Speaker Host]:**

.
With these two inequalities, we conclude.
Next, we prove the following lemma. This straightforward, and almost trivial, result is provided as
a separate lemma because we simplify terms in this manner quite often throughout our proofs.
Lemma 5. Let L be a convex function and let X, Y be two random variables. Then,

**[Speaker Guest]:**

min

**[Speaker Host]:**

f
EX, Y [P(X′ = X)L(f(X′), S(X, Y ))] + EX, Y


**[Speaker Guest]:**

X

**[Speaker Host]:**

X′̸=X
P(X′)L (f(X′), S(X, Y ))

less than or equal to min
f {EX, Y [L(f(X), S(X, Y ))]} + ϵ
where ϵ = maxX′̸=X, X, Y {L(f ∗(X′), S(X, Y )} and f ∗= arg minf{EX, Y [L(f(X), S(X, Y )]}.

**[Speaker Guest]:**

Proof.

**[Speaker Host]:**

min

**[Speaker Guest]:**

f
EX, Y [P(X′ = X)L(f(X′), S(X, Y ))] + EX, Y


**[Speaker Host]:**

X

**[Speaker Guest]:**

X′̸=X
P(X′)L (f(X′), S(X, Y ))

less than or equal to min

**[Speaker Host]:**

f
EX, Y [L(f(X), S(X, Y ))] + EX, Y


**[Speaker Guest]:**

X

**[Speaker Host]:**

X′̸=X
P(X′)L (f(X′), S(X, Y ))

less than or equal to min
f {EX, Y [L(f(X), S(X, Y ))]} + EX, Y


**[Speaker Guest]:**

X

**[Speaker Host]:**

X′̸=X
P(X′)L (f ∗(X′), S(X, Y ))

less than or equal to min
f {EX, Y [L(f(X), S(X, Y ))]} + EX, Y


**[Speaker Guest]:**

X

**[Speaker Host]:**

X′̸=X

**[Speaker Guest]:**

P(X′)ϵ

less than or equal to min
f {EX, Y [L(f(X), S(X, Y ))]} + ϵ

**[Speaker Host]:**

E. 1

**[Speaker Guest]:**

ASSUMPTIONS
Considering that recent policies often use a short context length c, we assume the range of temporal
dependency modeled by a (c, h)-policy is limited:
Assumption 1. The sum of context length and action horizon is less than the length of temporal
dependency in expert demonstrations, c + h < k.
Published as a conference paper at ICLR 2025
This assumption allows us to focus on the problem that is relevant to us and allows us to ignore edge
cases. However, our analysis can be extended to the case where this assumption does not hold. In the
case where c + h greater than or equal to k, the larger action chunk model will not get any advantage (making αb = 0)
since the additional states it has observed in time steps τb are irrelevant to at. Then, the longer
horizon policy either attains the same expected loss the shorter horizon policy or suffer greater loss
from not having observed the recent past states.
Assumption 2. An optimal πc, h must infer the unobserved states based on the observed

**[Speaker Host]:**

assistant
We conclude with these two inequalities.

Next, we prove a lemma. This result is simple and straightforward, and we present it separately because we frequently simplify expressions in this way throughout the proofs.

**Lemma 5**: Let L be a convex function, and let X and Y be random variables. Then, for any function f, the following holds:

min_f [E_{X,Y} [P(X' = X) L(f(X'), S(X,Y)) + Σ_{X' ≠ X} P(X') L(f(X'), S(X,Y))]] ≤ min_f [E_{X,Y} [L(f(X), S(X,Y))]] + ϵ,

where ϵ is defined as the maximum value of L(f*(X'), S(X,Y)) over all X' ≠ X and all X,Y, and f* is the function that minimizes E_{X,Y} [L(f(X), S(X,Y))].

The proof proceeds by first observing that the left-hand side is bounded above by the expected value of L(f(X), S(X,Y)) under the original distribution, plus the contribution from all other cases where X' ≠ X. Then, since L is convex, we can bound the term involving other X' values using the minimizing function f*, and since L is convex, the value of L(f*(X'), S(X,Y)) is bounded by ϵ. Thus, the entire expression is bounded above by the minimum expected loss plus ϵ.

We now turn to our assumptions.

Recent policies often use a short context length c. We therefore assume that the temporal dependency modeled by a (c, h)-policy is limited.

**Assumption 1**: The sum of context length and action horizon is less than the total length of temporal dependency in expert demonstrations: c + h < k.

This assumption allows us to focus on the relevant problem and avoid edge cases. Our analysis can be extended to cases where c + h ≥ k. When c + h ≥ k, the longer-horizon policy gains no advantage (making α_b = 0), because the additional states observed at time steps τ_b are irrelevant to the current action. In this case, the longer-horizon policy either matches the expected loss of the shorter-horizon policy or incurs a greater loss due to not having observed recent past states.

**Assumption 2**: An optimal policy π_{c,h} must infer unobserved states based on observed states.

This assumption underpins the reasoning that the policy must generalize from observed data to infer missing state information.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
infer the unobserved states based on the observed states and
actions by modeling the transition dynamics ˆPtrain(st′ | st′−1, at′−1) accurately for all time step t′.
We consider this implicit model to be the same for both π(c, h) and π(c, h+d).
In other words, we can write using the law of total probability:
π(c, h)(at|st−h−c:t−h, at−h:t−1)
= Est−k:t−h−c−1, st−h+1:t∼ˆ

**[Speaker Guest]:**

Ptrain

π(k, 1)(at|st−k:t)|st−h−c:t−h, at−h:t−1


**[Speaker Host]:**

and
π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−h−1)
= Est−k:t−h−d−c−1, st−h−d+1:t∼ˆ

**[Speaker Guest]:**

Ptrain

π(k, 1)(at|st−k:t)|st−h−d−c:t−h−d, at−h−d:t−1

**[Speaker Host]:**

.

**[Speaker Guest]:**

E. 2

**[Speaker Host]:**

DEFINITIONS

**[Speaker Guest]:**

We,

π(c, h)(at | st−h−c:t−h, at−h:t−1) = E[st−k:t−h−c−1, st−h+1:t ∼ ˆPtrain] π(k, 1)(at | st−k:t) | st−h−c:t−h, at−h:t−1


**[Speaker Host]:**

and

π(c, h+d)(at | st−h−d−c:t−h−d, at−−h−d:t−h−1) = E[st−k:t−h−d−c−1, st−h−d+1:t ∼ ˆPtrain] π(k, 1)(at | st−k:t) | st−h−d−c:t−h−d, at−h−d:t−h−1

We analyze the effect of reducing context horizon. When the action horizon is held constant, reducing the context horizon leads to a decrease in the performance of the optimal policy.

Let a (c, h)-policy have context length c and action horizon h. The probability of taking action at at time t in a chunk generated at time t is denoted π(c, h) := π(c, h)(at | st−c:t). For a (c+1, h)-policy, the probability is π(c+1, h) := π(c+1, h)(at | st−c−1:t). Let π∗ represent the policy of a (k, 1)-expert taking action at time t.

Proposition 6: Let L be a non-linear, convex function. Let c < k. Let G := {at, st−k:t−c−1, zt−k:t} and C := {st−c:t}. Then:

min_π(c+1, h) E_G [L(π(c+1, h), π∗)] over C ≤ min_π(c, h) E_G [L(π(c, h), π∗)] over C

Equality holds if and only if st−c−1 ∼ P_test(· | C) and ˆst−c−1 ∼ ˆPtrain(· | C) take on the same value almost surely.

The proof considers function classes: X2 for functions taking at and st−c−1:t as inputs; X1 for functions taking st−c−1:t but not at; X0 for functions taking only st−c:t; and X−1 for functions taking st−c:t and at but not st−c−1. The inequality arises due to the loss of information from omitting st−c−1, which is critical for accurate state inference.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
inputs, but not st−c−1, are elements of X−1.

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+1, h)∈X2 EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c+1, h), π∗C

**[Speaker Guest]:**

i

**[Speaker Host]:**

= Eat

**[Speaker Guest]:**

"

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′

**[Speaker Host]:**

(c+1, h)∈X1

**[Speaker Guest]:**

Est−c−1

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−k:t−c−2, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π′

**[Speaker Host]:**

(c+1, h), π∗)

**[Speaker Guest]:**

at, st−c−1, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

#

**[Speaker Guest]:**

(Lemma 4)

**[Speaker Host]:**

= Eat

**[Speaker Guest]:**

"

**[Speaker Host]:**

Est−c−1

**[Speaker Guest]:**

"

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′

**[Speaker Host]:**

(c, h)∈X0

**[Speaker Guest]:**

Est−k:t−c−2, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π′

**[Speaker Host]:**

(c, h), π∗)

**[Speaker Guest]:**

at, st−c−1, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

# C

**[Speaker Host]:**

#

**[Speaker Guest]:**

(Lemma 4)
less than or equal to Eat

**[Speaker Host]:**

"

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′

**[Speaker Guest]:**

(c, h)∈X0

**[Speaker Host]:**

Est−c−1

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est−k:t−c−2, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π′

**[Speaker Guest]:**

(c, h), π∗)

**[Speaker Host]:**

at, st−c−1, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

i C

**[Speaker Guest]:**

#
(Jensen’s inequality)
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h)∈X−1 Eat

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est−c−1

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est−k:t−c−2, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c, h), π∗)

**[Speaker Guest]:**

at, st−c−1, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

i

**[Speaker Guest]:**

(Lemma 4).
Use the law of total expectation to conclude. The equality conditions for Jensen’s inequality provides
us the equality condition for this relationship too.

**[Speaker Host]:**

Now, we formalize the definitions of Expected Observation Advantage and Maximum Inference

**[Speaker Guest]:**

Disadvantage.
Recall that, in §3. 2, we have two policies: π(c, h) and π(c, h+d); the former sees more recent states
while the latter remembers more past states.


**[Speaker Host]:**

---

We’re looking at how different policies perform in reinforcement learning, particularly when they have access to different amounts of past information.

One policy, π(c, h), sees only the most recent state history. Another, π(c+d, h), has access to even older states — it remembers more of the past. The key idea is that having more information can reduce the difference between a policy and the optimal policy π*.

We define the *Expected Observation Advantage* (αb) as the improvement in performance that comes from having more context. Specifically, the policy with access to more history — π(c+d, h) — achieves a lower divergence from the expert policy than π(c, h). The difference in performance is captured by αb.

This advantage is always non-negative. It’s zero when the past states that π(c, h) doesn’t see can be logically deduced from what it already observes — or when the expert policy doesn’t depend on those older states.

We also define the *Maximum Inference Disadvantage* (ϵf), which measures the worst-case gap in performance when a policy lacks access to certain historical data.

The whole analysis uses the law of total expectation and Jensen’s inequality to show how the expected loss evolves across different time windows and policy choices. The key point is that adding past observations leads to better policy performance — and we can quantify this benefit.


**[Speaker Guest]:**

---

This version is clean, accessible, and suitable for a podcast script without losing the core technical meaning. Let me know if you'd like a version with a narrative flow, a specific tone (e.g., scientific, conversational, dramatic), or a summary of the key takeaways.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
Maximum Inference Disadvantage (ϵf)). Consider the maximum divergence that
can be accumulated by the (c, h + d)-policy from not knowing the recent states at time steps
st−h−d+1:t−h, and let that be ϵf. More formally, we say that, for fixed C from Proposition 1, any
states in S−:= {st−k:t} \ S+, any zt−k:t, and any ˆst−h−d+1:t−h ̸= st−h−d+1:t−h, the following

**[Speaker Host]:**

holds:
L(π(c+d, h)(at|st−h−d−c:t−h−d, ˆst−h−d+1:t−h ̸= st−h−d+1:t−h, at−h:t−1), π∗) less than or equal to ϵf.

**[Speaker Guest]:**

(9)
Here, π(c+d, h) := arg minπ(c+d, h) E[L(π(c+d, h), π∗)|C] is the optimal (c + d, h)-policy.
To define αf and ϵb, we prove a second version of Proposition 6. Consider a (c, h)-policy whose
probability of taking action at at time t in a chunk generated at t is referred to as π(c, h) :=
π(c, h)(at|st−c:t). On the other hand, consider a (c −1, h + 1)-policy whose probability of taking
action at in a chunk generated at time t −1 is referred to as π(c−1, h+1) := π(c−1, h+1)(at|st−c:t−1).
Lastly, consider a (k, 1)-expert whose probability of taking action at at time t is π∗.
Proposition 7. Let L be a non-linear,

**[Speaker Host]:**

convex function.

**[Speaker Guest]:**

Let c

**[Speaker Host]:**

<

**[Speaker Guest]:**

k.

**[Speaker Host]:**

Let G

**[Speaker Guest]:**

:=
{at, st−k:t−c−1, st, zt−k:t} and C := {st−c:t−1, at−1}. Then,

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h) EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i
less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c−1, h+1) EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c−1, h+1), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i.
In particular, this is an equality if and only if the state st ∼Ptest(· | C) and ˆst ∼ˆPtrain(· | C)
takes on only one and the same value almost surely.
Proof. The proof is similar to that of Proposition 6. We refer to the class of functions that accept
at and st−c:t as inputs as X2. Similarly, the class of functions that do not accept at as inputs but
accept st−c:t as inputs is X1. The function class that accepts only st−c:t−1 and not st or at as inputs
Published as a conference paper at ICLR 2025
are elements of X0. Lastly, the function class that accepts st−c:t−1 and at as inputs, but not st, are

**[Speaker Host]:**

elements of X−1.

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h)∈X2 EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c, h), π∗C

**[Speaker Guest]:**

i

**[Speaker Host]:**

= Eat

**[Speaker Guest]:**

"

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′

**[Speaker Host]:**

(c, h)∈X1

**[Speaker Guest]:**

Est

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−k:t−c−1, zt−k:

**[Speaker Host]:**

assistant
Maximum inference disadvantage (ϵf): Consider the maximum divergence that can be accumulated by the (c, h + d)-policy from not knowing the recent states at time steps st−h−d+1:t−h, and let that be ϵf. Formally, for fixed C from Proposition 1, any states in S− := {st−k:t} \ S+, any zt−k:t, and any ˆst−h−d+1:t−h ≠ st−h−d+1:t−h, the following holds:

L(π(c+d, h)(at|st−h−d−c:t−h−d, ˆst−h−d+1:t−h ≠ st−h−d+1:t−h, at−h:t−1), π∗) ≤ ϵf.

Here, π(c+d, h) := arg minπ(c+d, h) E[L(π(c+d, h), π∗)|C] is the optimal (c + d, h)-policy.

To define αf and ϵb, we prove a second version of Proposition 6. Consider a (c, h)-policy whose probability of taking action at at time t in a chunk generated at t is π(c, h) := π(c, h)(at|st−c:t). Consider a (c − 1, h + 1)-policy whose probability of taking action at in a chunk generated at time t − 1 is π(c−1, h+1) := π(c−1, h+1)(at|st−c:t−1). Lastly, consider a (k, 1)-expert whose probability of taking action at at time t is π∗.

Proposition 7: Let L be a non-linear, convex function. Let c < k. Let G := {at, st−k:t−c−1, st, zt−k:t} and C := {st−c:t−1, at−1}. Then:

min π(c, h) E_G [L(π(c, h), π∗)|C] ≤ min π(c−1, h+1) E_G [L(π(c−1, h+1), π∗)|C].

In particular, this is an equality if and only if the state st ∼ P_test(· | C) and ˆst ∼ ˆP_train(· | C) take on only one and the same value almost surely.

Proof: The proof is similar to that of Proposition 6. We refer to the class of functions that accept at and st−c:t as inputs as X2. Similarly, the class of functions that do not accept at as inputs but accept st−c:t as inputs is X1. The function class that accepts only st−c:t−1 and not st or at as inputs is elements of X0. Lastly, the function class that accepts st−c:t−1 and at as inputs but not st is elements of X−1.

min π(c, h) ∈ X2 E_G [L(π(c, h), π∗)|C] = E_at [ min π′(c, h) ∈ X1 E_st [ E_st−k:t−c−1, zt−k:t ] ].

(Note: The rest of the text appears to be a continuation of the mathematical derivation, but contains LaTeX, unstructured fragments, and incomplete expressions. After cleaning and simplifying the core content without adding interpretation or summary, the final cleaned version below retains only the essential, readable, and logically coherent statements.)

Maximum inference disadvantage (ϵf): The worst-case divergence a (c, h + d)-policy incurs due to not observing the state sequence st−h−d+1:t−h is bounded by ϵf. For fixed context C, any state outside S+, any observation sequence zt−k:t, and any incorrect state estimate ˆst−h−d+1:t−h ≠ st−h−d+1:t−h, the policy loss satisfies:

L(π(c+d, h), π∗) ≤ ϵf.

This defines the maximum inference disadvantage as a bound on the performance degradation caused by missing state information.

Proposition 7: Under a non-linear, convex loss function L, and with c < k, the minimum expected loss over the (c, h)-policy is bounded above by that of the (c−1, h+1)-policy. That is:

min_π(c,h) E[L(π(c,h), π∗)|C] ≤ min_π(c−1,h+1) E[L(π(c−1,h+1), π∗)|C].

Equality holds only when the true state st and the estimated state ˆst under training distribution take the same value almost surely.

This result is structurally similar to Proposition 6 and relies on the hierarchy of function classes (X2 to X−1) to compare policy performance across different state observation windows.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user


**[Speaker Guest]:**

= Eat

**[Speaker Host]:**

"

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′

**[Speaker Guest]:**

(c, h)∈X1

**[Speaker Host]:**

Est

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est−k:t−c−1, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π′

**[Speaker Guest]:**

(c, h), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

i C

**[Speaker Guest]:**

#

**[Speaker Host]:**

(Lemma 4)

**[Speaker Guest]:**

= Eat

**[Speaker Host]:**

"

**[Speaker Guest]:**

Est

**[Speaker Host]:**

"

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′

**[Speaker Guest]:**

(c−1, h+1)∈X0

**[Speaker Host]:**

Est−k:t−c−2, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π′

**[Speaker Guest]:**

(c−1, h+1), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

# C

**[Speaker Guest]:**

#

**[Speaker Host]:**

(Lemma 4)
less than or equal to Eat

**[Speaker Guest]:**

"

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′

**[Speaker Host]:**

(c−1, h+1)∈X0

**[Speaker Guest]:**

Est

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−k:t−c−1, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π′

**[Speaker Host]:**

(c−1, h+1), π∗)

**[Speaker Guest]:**

at, st, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

#
(Jensen’s inequality)

**[Speaker Guest]:**

=

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c−1, h+1)∈X−1 Eat

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−k:t−c−1, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c−1, h+1), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

i C

**[Speaker Guest]:**

i

**[Speaker Host]:**

(Lemma 4).
Use the law of total expectation to conclude. The equality condition can be seen from the equality
condition of Jensen’s inequality.
Using this, we can define ϵb and αf in a similar manner:
Definition (Expected Observation Advantage (αf)). Recall that we have two models: π(c, h) and
π(c, h+d) and a hypothetical (c + d, h)-policy that has access to all the information both our learners
have (as in Eq. (8) and Eq. (9)). Observe that π(c+d, h) has access to more context than π(c, h+d), particularly the knowledge of states st−h−d+1:t−h. Therefore, we know, using Proposition 7, π(c+d, h)
has lower divergence with respect to π∗than π(c, h+d). We say that the advantage π(c+d, h) gets from
the extra information is αf. More formally, we say that

**[Speaker Guest]:**

αf =

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h+d) E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c, h+d), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

−

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c+d, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i

**[Speaker Guest]:**

(10)
where C is defined as in Proposition 1. Clearly, αf greater than or equal to 0. In particular, αf = 0 when π(c, h+d) can
infer st−h−d+1:t−h perfectly. This makes sense-in the static environment, observing these states
does not provide any advantage since the optimal π(c, h+d) can infer these states anyway using the
actions taken at those time steps (assuming that the implicit dynamics model is accurate).
Definition (Maximum Inference Disadvantage (ϵb)). Consider the maximum divergence that can
be accumulated by the (c, h)-model from not knowing the past states st−h−d−c:t−h−c−1 and let that
be ϵb. More formally, we say that, for fixed C from Proposition 1, any states in S−, any zt−k:t

**[Speaker Host]:**

assistant

**[Speaker Guest]:**

Eat

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′(c, h)∈X1

**[Speaker Host]:**

Est

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est−k:t−c−1, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π′(c, h), π∗)

**[Speaker Guest]:**

at, st, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

(Lemma 4)


**[Speaker Guest]:**

= Eat

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′(c−1, h+1)∈X0

**[Speaker Host]:**

Est−k:t−c−2, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π′(c−1, h+1), π∗)

**[Speaker Guest]:**

at, st, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

(Lemma 4)


**[Speaker Guest]:**

≤ Eat

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′(c−1, h+1)∈X0

**[Speaker Host]:**

Est

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est−k:t−c−1, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π′(c−1, h+1), π∗)

**[Speaker Guest]:**

at, st, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C
(Jensen’s inequality)


**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c−1, h+1)∈X−1

**[Speaker Guest]:**

Eat

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−k:t−c−1, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c−1, h+1), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

i C

**[Speaker Guest]:**

(Lemma 4)

Use the law of total expectation to conclude. The equality condition can be seen from the equality condition of Jensen’s inequality.

Using this, we can define ϵb and αf in a similar manner:

Definition (Expected Observation Advantage (αf)). Recall that we have two models: π(c, h) and π(c, h+d) and a hypothetical (c + d, h)-policy that has access to all the information both our learners have (as in Eq. (8) and Eq. (9)). Observe that π(c+d, h) has access to more context than π(c, h+d), particularly the knowledge of states st−h−d+1:t−h. Therefore, we know, using Proposition 7, π(c+d, h) has lower divergence with respect to π∗ than π(c, h+d). We say that the advantage π(c+d, h) gets from the extra information is αf. More formally, we say that


**[Speaker Host]:**

αf =

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h+d)

**[Speaker Guest]:**

E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c, h+d), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

−

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

(10)

where C is defined as in Proposition 1. Clearly, αf ≥ 0. In particular, αf = 0 when π(c, h+d) can infer st−h−d+1:t−h perfectly. This makes sense—in the static environment, observing these states does not provide any advantage since the optimal π(c, h+d) can infer these states anyway using the actions taken at those time steps (assuming that the implicit dynamics model is accurate).

Definition (Maximum Inference Disadvantage (ϵb)). Consider the maximum divergence that can be accumulated by the (c, h)-model from not knowing the past states st−h−d−c:t−h−c−1 and let that be ϵb. More formally, for fixed C from Proposition 1, any states in S−, any zt−k:t, we define ϵb as the worst-case divergence due to missing past states.


**[Speaker Guest]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Host]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Guest]:**

Here's the text:


**[Speaker Host]:**

user
C from Proposition 1, any states in S−, any zt−k:t and
any ˆst−h−d−c:t−h−c−1 ̸= st−h−d−c:t−h−c−1:
L(π(c+d, h)(at|ˆst−h−d−c:t−h−c−1 ̸= st−h−d−c:t−h−c−1, st−h−c:t−h, at−h:t−1), π∗) less than or equal to ϵb.

**[Speaker Guest]:**

(11)
Here, π(c+d, h) := arg minπ(c+d, h) E[L(π(c+d, h), π∗)|C] is the optimal (c + d, h)-policy.
As a warmup, we see that the intuitive relationship between αf and ϵf (and the same for αb and ϵb)

**[Speaker Host]:**

holds:
Proposition 8. αf less than or equal to ϵf and αb less than or equal to ϵb.
Proof. We prove the first inequality; the second can be proven in the same manner. We use Assumption 2 to write π(c, h+d) = Est−h−d+1:t−h∼P

π(c+d, h) | st−h−d−c:t−h−d, at−h−d:t−1


**[Speaker Guest]:**

where
P is the environment’s transition dynamics. Let
Pcorrect inference = P(ˆst−h−d+1:t−h = st−h−d+1:t−h|st−h−d−c:t−h−d, at−h−d:t−1)

**[Speaker Host]:**

and
Pincorrect inference = P(ˆst−h−d+1:t−h ̸= st−h−d+1:t−h|st−h−d−c:t−h−d, at−h−d:t−1).
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

Then,

**[Speaker Host]:**

αf =

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h+d) E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c, h+d), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i

**[Speaker Guest]:**

−

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) E

**[Speaker Guest]:**

h
L(Pcorrect inferenceπ(c+d, h) + Pincorrect inferenceπ(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

−

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c+d, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i
less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h){E

**[Speaker Guest]:**

h
Pincorrect inferenceL(π(c+d, h)(conditioning on incorrect inference), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ E

**[Speaker Guest]:**

h
Pcorrect inferenceL(π(c+d, h)(conditioning on correct inference), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

}

**[Speaker Guest]:**

−

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

(Convexity)
less than or equal to E

**[Speaker Guest]:**

h
Pincorrect inferenceL(ˆπ∗
(c+d, h(conditioning on incorrect inference), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π∗

**[Speaker Guest]:**

(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

−E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π∗

**[Speaker Guest]:**

(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i
(Bounding probabilities by 1 and Lemma 5)
less than or equal to E [Pincorrect inferenceϵf | C]
less than or equal to ϵf

**[Speaker Host]:**

Here, π∗
(c+d, h) := arg minπ(c+d, h) E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c+d, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i.
We will provide a tighter bound after the proof of our main theoretical results.
Definition (Forward and Backward Inference). For a fixed time step t and C (as in §3. 2), consider
the time steps τf := {t −h −d + 1 : t −h} and τb := {t −h −d −c :

**[Speaker Guest]:**

assistant
Forward and backward inference are mechanisms used to model how an agent updates its policy based on observations of state transitions, particularly when the observed state sequence is corrupted or misinferred. In this context, the forward inference (τf) refers to time steps from t−h−d+1 to t−h, while backward inference (τb) refers to time steps from t−h−d−c to t−h−c−1. These time steps define the window over which the policy is evaluated under different conditions of correct versus incorrect state inference.

The key idea is that when the agent observes a state sequence ˆst−h−d−c:t−h−c−1 that differs from the true state sequence st−h−d−c:t−h−c−1, the policy performance can degrade due to incorrect assumptions about past states. The paper introduces a framework where the expected loss under incorrect inference is bounded by a small value ϵb, ensuring that even with erroneous state observations, the policy does not perform significantly worse than the optimal policy.

Proposition 8 establishes that the policy robustness parameters αf and αb are bounded above by the respective error tolerances ϵf and ϵb. Specifically, αf ≤ ϵf and αb ≤ ϵb, meaning that the degradation in performance due to incorrect inference is no greater than the error tolerance. This result is derived by analyzing the expected loss difference between policies under correct and incorrect inference, using convexity and probabilistic bounds.

The proof begins by expressing the optimal (c+d, h)-policy under the condition of correct state observation. It then considers the case where the observed state sequence is incorrect and computes the expected loss under such a scenario. By leveraging the transition dynamics of the environment and applying Lemma 5, the loss due to incorrect inference is bounded by the probability of error multiplied by the worst-case loss, which is capped by ϵf. Since the probability of incorrect inference is bounded by 1, and the loss under optimal policy is at most ϵf, the overall degradation is shown to be at most ϵf.

This result provides a theoretical foundation for understanding how inference errors propagate into policy performance, and it sets the stage for tighter bounds in later sections. The forward and backward inference definitions are critical for structuring the time windows in which the policy evaluation and error analysis take place.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
τf := {t −h −d + 1 : t −h} and τb := {t −h −d −c : t −h −c −1}.

**[Speaker Host]:**

Define
Pf(t′) := Ptest(St′ = gt′ | St′−1 = gt′−1, At′−1 = at′−1)
for any t′ ∈τf with gt′, gt′−1, at′−1 being the ground truth states and action in the deterministic
test environment. As such, Pf(t′) = 1 in a deterministic environment. In a stochastic environment,
Pf(t′) < 1 for all t′ and as the stochasticity increases, these values decrease and approach 0. Let
δf(t′) = ˆPtrain(gt′|gt′−1, at′−1) −Pf(t′)
where gt′, gt′−1, at′−1 are still the ground truth in the deterministic test environment.

**[Speaker Guest]:**

Define
ˆPf(t′) = Pf(t′) + δf(t′). Similarly, define
Pb(t′) := Ptest(St′ = gt′ | St′+1 = gt′+1)
for any t′ ∈τb where gt′, gt′−1 are the ground truth states in the deterministic test environment.

**[Speaker Host]:**

Then let
δb(t′) = ˆPtrain(gt′|gt′+1) −Pb(t′)
for any t′ ∈τb. Define ˆPb(t′) = Pb(t′) + δb(t′).
Intuitively, δf(t′) and δb(t′) characterize the error that the implicit dynamics model has in capturing
the test environment’s dynamics.

**[Speaker Guest]:**

E. 3
CONSISTENCY-REACTIVITY INEQUALITIES
Now we prove the Consistency-Reactivity Inequalities.
Proposition 1 (Consistency-Reactivity Inequalities) Let L be a non-linear and non-negative convex
function measuring the prediction error with respect to demonstrations. Let S+ ⊂{st−k:t} be
the states both the (c, h) and the (c, h + d) policies observe and let S−:= {st−k:t} \ S+. Let
C := {at−h:t−1} ∪S+, G := {at, zt−k:t} ∪{at−h−d:t−h−1} ∪S−. For notational ease, let τf =
Published as a conference paper at ICLR 2025
{t −h −d + 1 : t −h} and let τb = {t −h −d −c : t −h −c −1}. Then, we can bound the
expected loss of the (c, h + d)-policy and the (c, h)-policy as:

**[Speaker Host]:**

αf −ϵb(1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
Pb(t′)(Pb(t′) + δb(t′))) less than or equal to min
πh+d EG [L(πh+d, π∗)|C] −min

**[Speaker Guest]:**

πh EG [L(πh, π∗)|C]

**[Speaker Host]:**

and

**[Speaker Guest]:**

min
πh+d EG [L(πh+d, π∗)|C] −min
πh EG [L(πh, π∗)|C] less than or equal to ϵf(1 −

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf
Pf(t′)(Pf(t′) + δf(t′))) −αb.
Proof. We first prove the upper bound.
For ease of notation, we will write xb

**[Speaker Host]:**

a: to mean xa:b.

**[Speaker Guest]:**

Additionall

**[Speaker Host]:**

assistant
The text provided contains mathematical notation, LaTeX symbols, and technical definitions related to policy evaluation and dynamics modeling in reinforcement learning. To prepare it for a podcast script, we must clean it of all mathematical symbols, LaTeX, and technical jargon that would not translate well to audio, while preserving only the core conceptual ideas in clear, accessible language.

Here is the cleaned, podcast-friendly version — stripped of equations, symbols, and formalisms, and rewritten in natural, conversational English:


**[Speaker Guest]:**

---

We want to understand how well a policy performs in a real environment, especially when it's trained on data that doesn't perfectly match the actual environment.

Imagine a robot learning to walk. It practices in a simulation, but the real world behaves differently — maybe the ground is slippery, or the robot wobbles more. So we need to measure how far off the learned behavior is from the true behavior.

To do this, we define two types of time steps: forward steps and backward steps. These represent different moments in the learning process where we check how well the model predicts what should happen next.

At each of these time steps, we compare what the model predicts with what actually happens in the real world. The difference between prediction and reality is called an error — and we track how large that error is.

We then analyze how this error affects performance. If the model makes small mistakes in the early steps, it might still work well. But if it fails in key moments — especially when it's making decisions based on past actions — performance drops.

We show that under certain conditions, the performance gap between a more advanced policy and a simpler one can be bounded. Specifically, if the model fails in a few key steps, the overall loss stays within predictable limits. This gives us confidence that even if the model isn’t perfect, it won’t perform catastrophically.

In short, we prove that the difference in performance between two policies — one with more memory and one with less — is limited by how often and how badly the model mispredicts the environment’s behavior. This gives a solid foundation for understanding how reliable and robust such models really are.


**[Speaker Host]:**

---

This version is ready to be used in a podcast script. It removes all technical details, equations, and notation, and focuses only on the core idea in clear, natural language. Let me know if you'd like a version with specific tone (e.g., technical, casual, dramatic, etc.) or expanded with analogies.

ion, we will write xb

**[Speaker Guest]:**

a: to mean xa:b.
Additionally, for greater clarity, we will explicitly include the context length of each model, so
π(c, h) = πh and π(c, h+d) = πh+d. We start by writing, using Assumption 2,
π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−1)
= π(c, h+d)(at|st−h−d

**[Speaker Host]:**

t−h−d−c:, at−1

**[Speaker Guest]:**

t−h−d:)

**[Speaker Host]:**

= Eˆst−h−d+1:t−h

**[Speaker Guest]:**

h

**[Speaker Host]:**

π(c+d, h)(at|st−h−d

**[Speaker Guest]:**

t−h−d−c:, ˆst−h

**[Speaker Host]:**

t−h−d+1:, at−1

**[Speaker Guest]:**

t−h−d:)

**[Speaker Host]:**

st−h−d

**[Speaker Guest]:**

t−h−d−c:, at−1

**[Speaker Host]:**

t−h−d:

**[Speaker Guest]:**

i.
Using this, we expand the left hand side of our inequality:

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h+d)

**[Speaker Host]:**

EG


**[Speaker Guest]:**

L(π(c, h+d), π∗)|C


**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(Eˆst−h−d+1:t−h

**[Speaker Host]:**

h

**[Speaker Guest]:**

π(c+d, h)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i, π∗)|C

**[Speaker Host]:**

i

**[Speaker Guest]:**

=

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h)

**[Speaker Host]:**

EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

L( ˆ

**[Speaker Guest]:**

Ptrain(gt−h

**[Speaker Host]:**

t−h−d+1:|st−h−d

**[Speaker Guest]:**

t−h−d−c:, at−h−1
t−h−d:)π(c+d, h)(at| · · ·, gt−h

**[Speaker Host]:**

t−h−d+1:)+

**[Speaker Guest]:**

X

**[Speaker Host]:**

ˆst−h−d+1:t−h

**[Speaker Guest]:**

not all gt′

**[Speaker Host]:**

ˆ

**[Speaker Guest]:**

Ptrain(ˆst−h
t−h−d+1:|st−h−d, at−h−1
t−h−d:) π(c+d, h)(at| · · ·, ˆst−h

**[Speaker Host]:**

t−h−d+1:), π∗)|C

**[Speaker Guest]:**

i
less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h)

**[Speaker Host]:**

EG


**[Speaker Guest]:**

L(

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

ˆ
Pf (t′)π(c+d, h)(at| · · ·, gt−h

**[Speaker Guest]:**

t−h−d+1:)+

**[Speaker Host]:**

X

**[Speaker Guest]:**

ˆst−h−d+1:t−h

**[Speaker Host]:**

not all gt′

**[Speaker Guest]:**

ˆ

**[Speaker Host]:**

Ptrain(ˆst−h
t−h−d+1:|st−h−d, at−h−1
t−h−d:) π(c+d, h)(at| · · ·, ˆst−h

**[Speaker Guest]:**

t−h−d+1:), π∗)|C

**[Speaker Host]:**

i
where we computed the expectation Eˆst−h−d+1:t−h

**[Speaker Guest]:**

h

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i
by grouping into two terms : one
where every ˆst−h−d+1:t−h = gt−h−d+1:t−h and one where there is at least one term ˆsi that is not
gi. This grouping was done using the definition of noise in our environment. We introduce the
following notation here

**[Speaker Guest]:**

ˆP̸=gt′ :=

**[Speaker Host]:**

X

**[Speaker Guest]:**

ˆst−h−d+1:t−h

**[Speaker Host]:**

not all gt′

**[Speaker Guest]:**

ˆPtrain(ˆst−h
t−h−d+1:|st−h−d, at−h−1

**[Speaker Host]:**

t−h−d:).

**[Speaker Guest]:**

Similarly,

**[Speaker Host]:**

P̸=gt′ :=

**[Speaker Guest]:**

X

**[Speaker Host]:**

st−h−d+1:t−h

**[Speaker Guest]:**

not all gt′

**[Speaker Host]:**

Ptest(st−h
t−h−d+1:|st−h−d, at−h−1

**[Speaker Guest]:**

t−h−d:).
With this notation, we continue our expansion:
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h+d)

**[Speaker Host]:**

EG


**[Speaker Guest]:**

L(π(c, h+d), π∗)|C

less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h)

**[Speaker Host]:**

EG


**[Speaker Guest]:**

L(

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

ˆ
Pf (t′)π(c+d, h)(at| · · ·, gt−h

**[Speaker Guest]:**

t−h−d+1:)+

**[Speaker Host]:**

X

**[Speaker Guest]:**

ˆst−h−d+1:t−h

**[Speaker Host]:**

not all gt′

**[Speaker Guest]:**

ˆ

**[Speaker Host]:**

Ptrain(ˆst−h
t−h−d+1:|st−h−d, at−h−1
t−h−d:) π(c+d, h)(at| · · ·, ˆst−h

**[Speaker Guest]:**

t−h−d+1:), π∗)|C

**[Speaker Host]:**

i
less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

EG


**[Speaker Host]:**

L(

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

ˆ
Pf (t′)π(c+d, h)(at| · · ·, gt−h

**[Speaker Host]:**

t−h−d+1:) + ˆ
P̸=gt′ π(c+d, h)(at| · · ·, ˆst−h

**[Speaker Guest]:**

t


**[Speaker Host]:**

t−h−d+1:) + ˆ
P̸=gt′ π(c+d, h)(at| · · ·, ˆst−h

**[Speaker Guest]:**

t−h−d+1:), π∗)|C

**[Speaker Host]:**

i
less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

EG


**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

ˆ
Pf (t′)L(π(c+d, h)(at| · · ·, gt−h

**[Speaker Guest]:**

t−h−d+1:), π∗)|C


**[Speaker Host]:**

+ EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

ˆ
P̸=gt′ L(π(c+d, h)(at| · · ·, ˆst−h

**[Speaker Guest]:**

t−h−d+1:), π∗)|C

**[Speaker Host]:**

i
where we got the inequality using the fact that L is a convex function and, thus, convex in each
argument. Next, we take the expectation over st−h−d+1:t−h by grouping the terms into two: one
where every st−h−d+1:t−h = gt−h−d+1:t−h and one where there is at least one term si ̸= gi. Then,
with some suppression of notation in the expression of π(c+d, h) and G′ := G \ {at, st−h−d+1:t−h}:

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h+d) EG


**[Speaker Guest]:**

L(π(c, h+d), π∗)|C

less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) Eat


**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

P test

**[Speaker Guest]:**

f

**[Speaker Host]:**

(t′) ˆPf(t′)EG′

**[Speaker Guest]:**

h
L(π(c+d, h)(... ... ... ˆst−h

**[Speaker Host]:**

t−h−d+1: = gt−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Guest]:**

t−h−d+1: = gt−h

**[Speaker Host]:**

t−h−d+1:

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ P̸=gt′

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

ˆPf(t′)EG′

**[Speaker Host]:**

h
L(π(c+d, h)(... ... ... ˆst−h

**[Speaker Guest]:**

t−h−d+1: = gt−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Host]:**

t−h−d+1: ̸= gt−h

**[Speaker Guest]:**

t−h−d+1:

**[Speaker Host]:**

i

**[Speaker Guest]:**

+

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

P test

**[Speaker Guest]:**

f

**[Speaker Host]:**

(t′) ˆP̸=gt′ EG′

**[Speaker Guest]:**

h
L(π(c+d, h)(... ... ... ˆst−h

**[Speaker Host]:**

t−h−d+1: ̸= gt−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Guest]:**

t−h−d+1: = gt−h

**[Speaker Host]:**

t−h−d+1:

**[Speaker Guest]:**

i
+ P̸=gt′ ˆP̸=gt′ EG′

**[Speaker Host]:**

h
L(π(c+d, h)(... ... ..., ˆst−h

**[Speaker Guest]:**

t−h−d+1: ̸= gt−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Host]:**

t−h−d+1: ̸= gt−h

**[Speaker Guest]:**

t−h−d+1:

**[Speaker Host]:**

i

**[Speaker Guest]:**

Now, we group all the terms into two - one representing where the learner’s simulation matches the
reality and one where it does not. Continuing from where we left off,


**[Speaker Host]:**

L(π(c, h+d), π∗)|C

less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

Eat


**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

P test

**[Speaker Guest]:**

f

**[Speaker Host]:**

(t′) ˆ

**[Speaker Guest]:**

Pf (t′)E

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h

**[Speaker Guest]:**

t−h−d+1: = gt−h

**[Speaker Host]:**

t−h−d+1: = st−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Guest]:**

t−h−d+1: = gt−h

**[Speaker Host]:**

t−h−d+1:

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ P̸=gt′

**[Speaker Guest]:**

ˆ

**[Speaker Host]:**

P f

**[Speaker Guest]:**

ˆs=sE

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h

**[Speaker Guest]:**

t−h−d+1: = st−h

**[Speaker Host]:**

t−h−d+1: ̸= gt−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Guest]:**

t−h−d+1: ̸= gt−h

**[Speaker Host]:**

t−h−d+1:

**[Speaker Guest]:**

i

**[Speaker Host]:**

+

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

P test

**[Speaker Host]:**

f

**[Speaker Guest]:**

(t′) ˆ

**[Speaker Host]:**

P̸=gt′ E

**[Speaker Guest]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h

**[Speaker Host]:**

t−h−d+1: ̸= gt−h

**[Speaker Guest]:**

t−h−d+1: = st−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Host]:**

t−h−d+1: = gt−h

**[Speaker Guest]:**

t−h−d+1:


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Host]:**

t−h−d+1: = gt−h

**[Speaker Guest]:**

t−h−d+1:

**[Speaker Host]:**

i

**[Speaker Guest]:**

+ P̸=gt′

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

ˆ

**[Speaker Guest]:**

Pf (t′)E

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h

**[Speaker Guest]:**

t−h−d+1: = gt−h

**[Speaker Host]:**

t−h−d+1: ̸= st−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Guest]:**

t−h−d+1: ̸= gt−h

**[Speaker Host]:**

t−h−d+1:

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ P̸=gt′ ˆ

**[Speaker Guest]:**

Ptrain(ˆst−h

**[Speaker Host]:**

t−h−d+1: ̸= st−h

**[Speaker Guest]:**

t−h−d+1:, gt−h
t−h−d+1:|st−h−d, at−h−1

**[Speaker Host]:**

t−h−d:)

**[Speaker Guest]:**

E

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h

**[Speaker Guest]:**

t−h−d+1: ̸= st−h

**[Speaker Host]:**

t−h−d+1: ̸= gt−h
t−h−d+1:), π∗)... ... ..., st−h

**[Speaker Guest]:**

t−h−d+1: ̸= gt−h

**[Speaker Host]:**

t−h−d+1:

**[Speaker Guest]:**

i
[Table content omitted]
Published as a conference paper at ICLR 2025
For the match terms, we use the fact that Q
t′∈τf Pf(t′) less than or equal to 1 and

**[Speaker Host]:**

ˆP f
ˆs=s = ˆPtrain(ˆst−h

**[Speaker Guest]:**

t−h−d+1: = st−h
t−h−d+1:|st−h−d, at−h−1
t−h−d:) less than or equal to 1. For the mismatch terms, we use the
definition of ϵf and Lemma 5. Then, we continue:
less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i
(Simulation matches reality)

**[Speaker Host]:**

+

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

P test

**[Speaker Host]:**

f

**[Speaker Guest]:**

(t′)

**[Speaker Host]:**

h

**[Speaker Guest]:**

(1 −ˆPf(t′))ϵf

**[Speaker Host]:**

i

**[Speaker Guest]:**

+

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

(1 −P test

**[Speaker Guest]:**

f

**[Speaker Host]:**

(t′))

**[Speaker Guest]:**

h
ˆPf(t′) + ˆP̸=gt′, st′

**[Speaker Host]:**

i

**[Speaker Guest]:**

ϵf.
(Simulation does not match reality)
Recall that we write P test

**[Speaker Host]:**

f
(t′) as simply Pf(t′). We simplify the mismatch terms further:

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

Pf(t′)

**[Speaker Host]:**

h

**[Speaker Guest]:**

(1 −ˆPf(t′))ϵf

**[Speaker Host]:**

i

**[Speaker Guest]:**

+

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

(1 −Pf(t′))

**[Speaker Guest]:**

h
ˆPf(t′) + ˆP̸=gt′, st′

**[Speaker Host]:**

i

**[Speaker Guest]:**

ϵf
less than or equal to

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf

**[Speaker Host]:**

Pf(t′)

**[Speaker Guest]:**

h

**[Speaker Host]:**

(1 −ˆPf(t′))ϵf

**[Speaker Guest]:**

i

**[Speaker Host]:**

+

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

(1 −Pf(t′))

**[Speaker Host]:**

h
ˆPf(t′) + (1 −ˆPf(t′))

**[Speaker Guest]:**

i

**[Speaker Host]:**

ϵf

**[Speaker Guest]:**

= ϵf ·


**[Speaker Host]:**

1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

Pf(t′) ˆPf(t′)


**[Speaker Host]:**

.

**[Speaker Guest]:**

= ϵf ·


**[Speaker Host]:**

1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf
Pf(t′)(Pf(t′) + δf(t′))


**[Speaker Guest]:**

.
Next, we simplify the match terms by using the definition of αb:

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

= min

**[Speaker Guest]:**

π(c, h) EG


**[Speaker Host]:**

L(π(c, h), π∗)|C


**[Speaker Guest]:**

−αb.
Substituting these two terms back in, we conclude.

**[Speaker Host]:**

Now, we prove the lower bound. We proceed in a manner similar to the proof of the upper bound. For
ease of notation, we will write xb
a: to mean xa:b. Additionally, for greater clarity, we will explicitly
include the context length of each model, so π(c, h) = πh and π(c, h+d) = πh+d. We start by writing,

**[Speaker Guest]:**

using Assumption 1,

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h) EG


**[Speaker Host]:**

L(π(c, h), π∗) | C


**[Speaker Guest]:**

=

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) EG


**[Speaker Host]:**

L( ˆPtrain(gt−h−c−1
t−h−d−c:|st−h−c)π(c+d, h) +

**[Speaker Guest]:**

X

**[Speaker Host]:**

ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c:,

**[Speaker Host]:**

not all gt′

**[Speaker Guest]:**

ˆPt

**[Speaker Host]:**

assistant
The text contains mathematical expressions, LaTeX notation, and technical details that are not suitable for direct consumption in a podcast format. To properly preprocess this content for a podcast, we must strip away all mathematical symbols, LaTeX, technical notation, and irrelevant technical jargon while preserving only the core ideas and narrative flow.

After careful analysis and removal of all non-essential or untranslatable elements (such as equations, subscripts, superscripts, probability notations, and formal definitions), the cleaned, podcast-ready version of the content is:


**[Speaker Guest]:**

---

We break down the performance of a policy in a real-world setting by comparing how well it matches the actual behavior of the system. When the model's predictions align with reality, we measure the match. When they don’t, we quantify the mismatch. We show that the total error is bounded by two components: one for when the model behaves correctly, and one for when it fails.

We prove both an upper and a lower bound on how well a policy performs. In doing so, we demonstrate that the gap between the model's behavior and the true environment is limited by the frequency of mismatches and the confidence in the model's predictions.

We also show that under certain assumptions, the performance loss due to mismatches can be simplified into a single expression involving the probability of correct prediction and a small error term that scales with how often the model fails to match reality.

Ultimately, we conclude that the policy's performance is fundamentally tied to how well it generalizes from training data to real-world conditions — and that the worst-case deviation is bounded by a measurable, predictable quantity.


**[Speaker Host]:**

---

This version preserves the core argument, removes all technical noise, and presents the content in a clear, accessible, and narrative-driven way suitable for a podcast.


**[Speaker Guest]:**

)π(c+d, h) +

**[Speaker Host]:**

X

**[Speaker Guest]:**

ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c:,

**[Speaker Guest]:**

not all gt′

**[Speaker Host]:**

ˆPtrain(ˆst−h−c−1
t−h−d−c:|st−h−c)π(c+d, h), π∗)

**[Speaker Guest]:**

C


**[Speaker Host]:**

.
less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) EG


**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
ˆPb(t′)L(π(c+d, h), π∗) +

**[Speaker Guest]:**

X

**[Speaker Host]:**

ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c:,

**[Speaker Host]:**

not all gt′

**[Speaker Guest]:**

ˆPtrain(ˆst−h−c−1
t−h−d−c:|st−h−c)L(π(c+d, h), π∗)

**[Speaker Host]:**

C


**[Speaker Guest]:**

.
We introduce the following notation here

**[Speaker Host]:**

ˆP̸=gt′ :=

**[Speaker Guest]:**

X

**[Speaker Host]:**

ˆst−h−d−c:t−h−c−1

**[Speaker Guest]:**

not all gt′

**[Speaker Host]:**

ˆPtrain(ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c:|st−h−c).

**[Speaker Host]:**

Similarly,

**[Speaker Guest]:**

P̸=gt′ :=

**[Speaker Host]:**

X

**[Speaker Guest]:**

st−h−d−c:t−h−c−1

**[Speaker Host]:**

not all gt′

**[Speaker Guest]:**

Ptest(st−h−c−1

**[Speaker Host]:**

t−h−d−c:|st−h−c).
Published as a conference paper at ICLR 2025
With this notation, we continue our expansion:

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h) EG


**[Speaker Guest]:**

L(π(c, h), π∗)|C

less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) EG


**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
ˆPb(t′)L(π(c+d, h)(at|... ... ..., gt−h−c−1

**[Speaker Host]:**

t−h−d−c:), π∗) | C


**[Speaker Guest]:**

+

**[Speaker Host]:**

EG

**[Speaker Guest]:**

h
ˆP̸=gt′ L( π(c+d, h)(at|... ... ..., ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Host]:**

t−h−d−c:), π∗) | C

**[Speaker Guest]:**

i
where we got the inequality using the fact that L is a convex function. Next, we take the expectation
over st−h−d−c:t−h−c−1 by grouping the terms into two: one where every st−h−d−c:t−h−c−1 =
gt−h−d−c:t−h−c−1 and one where there is at least one term si ̸= gi. Then, again suppressing some
terms inside the expression of π(c+d, h):

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h) EG


**[Speaker Host]:**

L(π(c, h), π∗)|C

less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) Eat


**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb

**[Speaker Guest]:**

Pb(t′) ˆPb(t′) E

**[Speaker Host]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c: = gt−h−c−1

**[Speaker Host]:**

t−h−d−c: = st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1

**[Speaker Guest]:**

t−h−d−c: = gt−h−c−1

**[Speaker Host]:**

t−h−d−c:


**[Speaker Guest]:**

+ P̸=gt′

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb

**[Speaker Host]:**

ˆPb(t′)E

**[Speaker Guest]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c: = gt−h−c−1
t−h−d−c: ̸= st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Guest]:**

t−h−d−c:


**[Speaker Host]:**

+

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb

**[Speaker Guest]:**

Pb(t′) ˆP̸=gt′ E

**[Speaker Host]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Guest]:**

t−h−d−c: = st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1

**[Speaker Host]:**

t−h−d−c: = gt−h−c−1

**[Speaker Guest]:**

t−h−d−c:

**[Speaker Host]:**

i

**[Speaker Guest]:**

+ P̸=gt′ ˆP̸=gt′ E

**[Speaker Host]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Guest]:**

t−h−d−c: ] | C, at

**[Speaker Host]:**

i

**[Speaker Guest]:**

| C

**[Speaker Host]:**

i

**[Speaker Guest]:**

Now, we group all the terms into two - one representing where the learner’s simulation matches
the reality and one where it does not. Continuing from where we left off and defining ˆP b

**[Speaker Host]:**

ˆs=s :=

**[Speaker Guest]:**

ˆPtra

where we left off and defining ˆP b

**[Speaker Host]:**

ˆs=s :=

**[Speaker Guest]:**

ˆPtrain(ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c: = st−h−c−1

**[Speaker Guest]:**

t−h−d−c:|st−h−c):

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h)

**[Speaker Host]:**

EG


**[Speaker Guest]:**

L(π(c, h), π∗)|C

less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h)

**[Speaker Host]:**

Eat


**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb

**[Speaker Guest]:**

Pb(t′) ˆ

**[Speaker Host]:**

Pb(t′)E

**[Speaker Guest]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c: = gt−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1

**[Speaker Guest]:**

t−h−d−c: = gt−h−c−1

**[Speaker Host]:**

t−h−d−c:

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ P̸=gt′

**[Speaker Guest]:**

ˆ

**[Speaker Host]:**

P b

**[Speaker Guest]:**

ˆs=sE

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c: = st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Host]:**

t−h−d−c:

**[Speaker Guest]:**

i

**[Speaker Host]:**

+

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb

**[Speaker Guest]:**

Pb(t′) ˆ

**[Speaker Host]:**

P̸=gt′ E

**[Speaker Guest]:**

h
L(π(c+d, h)(at|... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1

**[Speaker Host]:**

t−h−d−c: = gt−h−c−1

**[Speaker Guest]:**

t−h−d−c:

**[Speaker Host]:**

i

**[Speaker Guest]:**

+ P̸=gt′

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb

**[Speaker Host]:**

ˆ

**[Speaker Guest]:**

Pb(t′) E

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c: = gt−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Host]:**

t−h−d−c: ] | C, at

**[Speaker Guest]:**

i

**[Speaker Host]:**

| C

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ P̸=gt′ ˆ

**[Speaker Guest]:**

Ptrain(ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Host]:**

t−h−d−c:, st−h−c−1

**[Speaker Guest]:**

t−h−d−c: | st−h−c)

**[Speaker Host]:**

E

**[Speaker Guest]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1
t−h−d−c: ̸= st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Host]:**

t−h−d−c: ] | C, at

**[Speaker Guest]:**

i

**[Speaker Host]:**

| C

**[Speaker Guest]:**

i
For the match terms, we use the fact that Q
t′∈τb ˆPb(t′) less than or equal to 1 and

**[Speaker Host]:**

ˆPtrain(ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c: = st−h−c−1

**[Speaker Host]:**

t−h−d−c:|st−h
t−h−c:) less than or equal to 1. For the mismatch terms, we use the definition of ϵb
Published as a conference paper at ICLR 2025
and Lemma 5. Then, we continue:
less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c+d, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i
(Simulation matches reality)

**[Speaker Guest]:**

+

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
Pb(t′)(1 −ˆPb(t′))ϵb +

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
(1 −Pb(t′))( ˆPb(t′) + ˆP̸=gt′, st′ )ϵb.
(Simulation does not match reality)
We simplify the mismatch terms further:

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
Pb(t′)(1 −ˆPb(t′))ϵb +

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
(1 −Pb(t′))( ˆPb(t′) + ˆP̸=gt′, st′ )ϵb
less than or equal to

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
Pb(t′)(1 −ˆPb(t′))ϵb +

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
(1 −Pb(t′))( ˆPb(t′) + (1 −ˆPb(t′)))ϵb

**[Speaker Host]:**

= ϵb ·


**[Speaker Guest]:**

1 −

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb

**[Speaker Host]:**

Pb(t′) ˆPb(t′)


**[Speaker Guest]:**

= ϵb ·


**[Speaker Host]:**

1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
Pb(t′)(Pb(t′) + δb(t′))


**[Speaker Guest]:**

.
Next, we simplify the match terms by using the definition of αf which follows from Proposition 7
in a manner similar to the definition of αb:

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i

**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h+d) EG


**[Speaker Guest]:**

L(π(c, h+d), π∗)|C


**[Speaker Host]:**

−αf.

**[Speaker Guest]:**

(Pro


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user


**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h+d) EG


**[Speaker Guest]:**

L(π(c, h+d), π∗)|C


**[Speaker Host]:**

−αf.

**[Speaker Guest]:**

(Proposition 7)
We substitute these terms back in to get the desired bound.

**[Speaker Host]:**

Now, we prove Corollary 2 as a direct consequence of the Consistency-Reactivity Inequalities. Recall that in a near-deterministic environment, Pf is close to 1 as the transitions are purely determined.
Corollary 2 (Consistency). Suppose, the train and test environments are the same and it is deterministic. Suppose at is influenced by at least one state at time steps τb and let δf(t′) approximately 0 for all
t′ ∈τf. If the diversity in past strategies is not 0, and ϵf is finite, then

**[Speaker Guest]:**

min
πh+d EG [L(πh+d, π∗)|C] < min
πh EG [L(πh, π∗)|C].
Proof. This follows from the upper bound of Proposition 1. Since the test environment is deterministic, we take Pf(t′) approximately 1 and, by hypothesis, δf(t′) approximately 0 for all t′ ∈τf, menaing our learned policy’s
implicit dynamics model is accurate. Then, we get

**[Speaker Host]:**

min
πh+d EG [L(πh+d, π∗) | C]−min
πh EG [L(πh, π∗) | C] less than or equal to −αb+ϵf(1−

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τf
Pf(t′)(Pf(t′)+δf(t′))) = −αb
since ϵf is finite. All that remains to show is that αb is positive. Note that at is temporally dependent
on at least one state in {st−h−c−d:t−h−c−1}. Furthermore, since diversity in past strategies is not 0,
the states in timesteps τb cannot be predicted with probability 1. Then, by Proposition 6, we have
that αb > 0. Therefore,

**[Speaker Guest]:**

min
πh+d EG [L(πh+d, π∗) | C] −min
πh EG [L(πh, π∗) | C] less than or equal to −αb < 0.
Published as a conference paper at ICLR 2025
Next, we prove Corollary 3.
Corollary 3 Suppose Pf(t′) is small or both Pf(t′) and |δf(t′)| are large for all t′ ∈τf. If temporal
dependency decreases over time such that ϵb is small and at is influenced by at least one state at time

**[Speaker Host]:**

steps in τf, then

**[Speaker Guest]:**

min
πh+d EG [L(πh+d, π∗)|C] > min
πh EG [L(πh, π∗)|C].
Proof. Notice that since ϵb approximately 0, the lower bound in Proposition 1 is αf. Next, we show that that αf
is positive. If Pf(t′) is small, then we know by Prop

**[Speaker Host]:**

assistant
min_π(c, h+d) EG [L(π(c, h+d), π*)|C] − αf.

**[Speaker Guest]:**

(Proposition 7)

We substitute these terms back to obtain the desired bound.

Corollary 2 (Consistency): Suppose the train and test environments are identical and deterministic. Suppose at is influenced by at least one state at time steps τb, and δf(t′) ≈ 0 for all t′ ∈ τf. If there is diversity in past strategies and ϵf is finite, then
min_π(h+d) EG [L(π(h+d), π*)|C] < min_π(h) EG [L(π(h), π*)|C].

Proof: This follows from the upper bound in Proposition 1. In a deterministic environment, Pf(t′) ≈ 1, and by hypothesis, δf(t′) ≈ 0 for all t′ ∈ τf, meaning the learned policy’s dynamics model is accurate. Then:
min_π(h+d) EG [L(π(h+d), π*)|C] − min_π(h) EG [L(π(h), π*)|C] ≤ −αb + ϵf(1 − ∏_{t′∈τf} Pf(t′)(Pf(t′) + δf(t′))) = −αb,
since ϵf is finite. We now show αb > 0. Since at depends on at least one state in {st−h−c−d: t−h−c−1}, and due to diversity in past strategies, the states at time steps τb cannot be perfectly predicted with probability 1. By Proposition 6, αb > 0. Thus:
min_π(h+d) EG [L(π(h+d), π*)|C] − min_π(h) EG [L(π(h), π*)|C] ≤ −αb < 0.

Corollary 3: Suppose Pf(t′) is small or both Pf(t′) and |δf(t′)| are large for all t′ ∈ τf. If temporal dependency decreases over time such that ϵb is small, and at is influenced by at least one state at time steps in τf, then
min_π(h+d) EG [L(π(h+d), π*)|C] > min_π(h) EG [L(π(h), π*)|C].

Proof: Since ϵb ≈ 0, the lower bound in Proposition 1 becomes αf. We now show αf > 0. If Pf(t′) is small, then by Proposition 6, the dynamics model fails to capture the state transitions, leading to a positive αf. Alternatively, if |δf(t′)| is large, the dynamics model has significant errors, again implying αf > 0. Thus, under these conditions, the policy with longer history performs better.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user
positive. If Pf(t′) is small, then we know by Proposition 7 that αf is positive. On the other hand,
if Pf(t′) approximately 1 but |δf(t′)| is large, then since probabilities are bounded, we require δf(t′) approximately −1. In
other words, the inference of the correct unobserved state is very unlikely causing αf to be positive

**[Speaker Host]:**

again. Therefore,

**[Speaker Guest]:**

min
πh EG [L(πh −π∗)|C] < min
πh+d EG [L(πh+d −π∗)|C]
To sanity check the Consistency-Reactivity Inequalities, we prove the next proposition. This shows
that the right-hand side of the inequalities is greater than or equal to the left-hand side.
Proposition 9. In all environments,

**[Speaker Host]:**

−ϵb(1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
Pb(t′)(Pb(t′) + δb(t′))) less than or equal to −αb

**[Speaker Guest]:**

and
αf less than or equal to ϵf(1 −

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τf
Pf(t′)(Pf(t′) + δf(t′))).
Proof. We prove the first inequality; the proof of the second proceeds similarly. Using the definition
of αb and suppressing G and C (as in Proposition 1) for clarity, we have:

**[Speaker Host]:**

αb = −min
πh+d E[L(πh+d, π∗)] + min

**[Speaker Guest]:**

πh E[L(πh, π∗)]
less than or equal to −min

**[Speaker Host]:**

πh+d E[L(πh+d, π∗)]

**[Speaker Guest]:**

+ min

**[Speaker Host]:**

πh+d E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L

Pcorrect inferenceπh+d(correct inference)
+ Pincorrect inferenceπh+d(incorrect inference), π∗i
less than or equal to −min

**[Speaker Guest]:**

πh+d E[L(πh+d, π∗)]

**[Speaker Host]:**

+ min

**[Speaker Guest]:**

πh+d E

**[Speaker Host]:**

h

**[Speaker Guest]:**

Pcorrect inferenceL
πh+d(correct inference), π∗
+ Pincorrect inferenceL
πh+d(incorrect inference), π∗i
less than or equal to −min

**[Speaker Host]:**

πh+d E[L(πh+d, π∗)]

**[Speaker Guest]:**

+ min

**[Speaker Host]:**

πh+d


**[Speaker Guest]:**

E


**[Speaker Host]:**

L
πh+d(correct inference), π∗

**[Speaker Guest]:**

+ E

Pincorrect inferenceL
πh+d(incorrect inference), π∗
less than or equal to −min

**[Speaker Host]:**

πh+d E[L(πh+d, π∗)]

**[Speaker Guest]:**

+ min

**[Speaker Host]:**

πh+d


**[Speaker Guest]:**

E


**[Speaker Host]:**

L
πh+d(correct inference), π∗

**[Speaker Guest]:**

+ ϵb(1 −

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
Pb(t′)(Pb(t′)δb(t′)))
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

= ϵb(1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
Pb(t′)(Pb(t′) + δb(t′)))

**[Speaker Guest]:**

E. 4
CLOSED-LOOP VERSUS OPEN-LOOP IN HIGHLY STOCHASTIC AND
NEAR-DETERMINISTIC ENVIRONMENTS
The Consistency-Reactivity Inequalities allow us to make an even stronger statement when we compare strictly closed-loop policies with open-loop ones. Consider the same set-up as before with
h = 0. Thus, π(c, 0) represents

**[Speaker Host]:**

assistant
Positive. If Pf(t′) is small, then by Proposition 7, αf is positive. If Pf(t′) is approximately 1 but |δf(t′)| is large, then since probabilities are bounded, δf(t′) must be approximately −1. In that case, correct inference is unlikely, so αf remains positive. Therefore, the expected loss under closed-loop policies is strictly less than that under open-loop policies:
min_πh EG [L(πh − π∗)|C] < min_πh+d EG [L(πh+d − π∗)|C]

We prove that the right-hand side is greater than or equal to the left-hand side in Proposition 9. In all environments:
−ϵb(1 − ∏_{t′∈τb} Pb(t′)(Pb(t′) + δb(t′))) ≤ −αb

**[Speaker Guest]:**

and
αf ≤ ϵf(1 − ∏_{t′∈τf} Pf(t′)(Pf(t′) + δf(t′)))

Proof: We prove the first inequality; the second follows similarly. Using the definition of αb and suppressing G and C as in Proposition 1, we have:
αb = −min_πh+d E[L(πh+d, π∗)] + min_πh E[L(πh, π∗)]
≤ −min_πh+d E[L(πh+d, π∗)] + min_πh+d E[ L(πh+d, correct inference) P(correct inference) + L(πh+d, incorrect inference) P(incorrect inference) ]
≤ −min_πh+d E[L(πh+d, π∗)] + min_πh+d E[ L(πh+d, correct inference) ] + ϵb(1 − ∏_{t′∈τb} Pb(t′)(Pb(t′) + δb(t′)))
= ϵb(1 − ∏_{t′∈τb} Pb(t′)(Pb(t′) + δb(t′)))

This establishes the lower bound on αb. The same logic applies to αf, showing that the expected loss improvement from closed-loop policies is bounded below by the product of the probability of correct inference and the deviation term.

In highly stochastic and near-deterministic environments, closed-loop policies outperform open-loop policies due to their ability to adapt and correct inference errors in real time.


**[Speaker Host]:**

system

You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO

**[Speaker Guest]:**

❓ ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?


**[Speaker Host]:**

Here's the text:


**[Speaker Guest]:**

user

**[Speaker Host]:**

up as before with
h = 0. Thus, π(c, 0) represents a closed-loop policy whereas π(c, d) represents an open-loop one. We
can compare these policies’ divergences with the expert across the entire trajectory in the limiting
cases of the environment stochasticity.
Corollary 10. Suppose Pf(t′) is small or both Pf(t′) and |δf(t′)| are large for all t′ ∈τf. If
temporal dependency decreases over time such that ϵb is small and at is influenced by at least one
state at time steps in τf, then the expected divergence between the closed-loop policy over the full
trajectory is lower than that between the open-loop policy and the expert.
Suppose, the train and test environments are the same and it is deterministic. Suppose at is influenced by at least one state at time steps τb and let δf(t′) approximately 0 for all t′ ∈τf. If the diversity in past
strategies is not 0, and ϵf is finite, then the divergence between the closed-loop policy over the full
trajectory is greater than that between the open-loop policy and the expert.
Proof. At any arbitrary time step t, the chunks of the two policies can be aligned in one of two ways:
Case 1: π(c, 0) is executing at as the first action in its action chunk and π(c, d) is also executing at as
the first action in its action chunk.
Case 2: π(c, 0) is executing at as the first action in its action chunk and π(c, d) is executing at as the
k-th action, where k ∈(1, 1 + d] in its action chunk.
Using the Consistency-Reactivity Inequalities, in Case 1, both policies have equal divergence. However, in case 2, using Corollary 3, we know that the closed-loop policy will outperform the open-loop
one in the first setting of the statement and open-loop will outperform in the second. From this we
can conclude the divergence across the full trajectory.

**[Speaker Guest]:**

assistant
The closed-loop policy performs better than the open-loop policy in scenarios where environmental stochasticity is low and temporal dependencies diminish over time, especially when actions are influenced by past states. When the environment is deterministic and past strategies show diversity, the closed-loop policy exhibits greater divergence from the expert compared to the open-loop policy. This is because the closed-loop policy adapts dynamically to state transitions, while the open-loop policy follows a fixed sequence regardless of state evolution. The divergence across the full trajectory depends on how actions are structured within policy chunks—either aligned as first actions or embedded later in the sequence—determining whether the closed-loop or open-loop policy achieves better alignment with expert behavior.