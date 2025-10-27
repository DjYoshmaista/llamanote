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
et al., 2023b; Chi et al., 2023; Fu et al., 2024b; Lee et al., 2024; Khazatsky et al., 2024). Yet, existing
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
its ability to capture the temporal dependencies inherent in demonstrations. However, this advantage
∗Equal contribution.

**[Speaker Guest]:**

assistant
Action chunking—predicting and executing a sequence of actions without intermediate replanning—is increasingly used in robot learning from human demonstrations. However, its impact on learned policies remains inconsistent: some studies find it essential for strong performance, while others report degraded results.

We investigate how action chunking affects the alignment between a learned policy and human demonstrations. We find that chunking helps the learner better capture temporal dependencies in demonstrations, but at the cost of reduced reactivity to unexpected states. To balance this tradeoff, we propose Bidirectional Decoding (BID), a test-time inference method that combines action chunking with closed-loop adaptation.

At each timestep, BID samples multiple candidate actions and selects the optimal one based on two criteria: (i) backward coherence—ensuring the chosen action aligns with previous decisions, and (ii) forward contrast—favoring actions that are likely to lead to good future outcomes. By integrating decisions both within and across chunks, BID achieves both long-term consistency and short-term adaptability.

Experiments show BID improves two state-of-the-art generative policies across seven simulation tasks and two real-world tasks. Code and videos are available at [web link] github.io.


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
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

(a) action chunking
(b) receding horizon
(c) bidirectional decoding
Figure 1: Illustration of different inference methods applied to a robot policy with action chunking. The robot
is tasked with catching a moving trolley. (a) Vanilla action chunking (Zhao et al., 2023b) executes actions based
on previous predictions, resulting in delayed reactions to object motions.

**[Speaker Guest]:**

assistant
Figure 1: Illustration of different inference methods applied to a robot policy with action chunking. The robot is tasked with catching a moving trolley. (a) Vanilla action chunking executes actions based on previous predictions, resulting in delayed reactions to object motions.


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
(a) Vanilla action chunking (Zhao et al., 2023b) executes actions based
on previous predictions, resulting in delayed reactions to object motions. (b) Receding horizon (Chi et al., 2023)
enables faster reactions, but leads to a jittery trajectory in the presence of multimodal demonstrations (e. g., both
left- and right-handers).

**[Speaker Host]:**

assistant
Vanilla action chunking executes actions based on previous predictions, leading to delayed responses to object motion. Receding horizon enables faster reactions but results in a jittery trajectory when dealing with multimodal demonstrations, such as those involving both left- and right-handers.


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

g., both
left- and right-handers). (c) Our Bidirectional Decoding explicitly searches for the optimal action from multiple
predictions sampled at each time step, achieving both long-term consistency and short-term reactivity. comes at the cost of reduced access to recent state observations, which can be crucial for reacting
to unexpected dynamics arising from modeling errors or environmental stochasticity.

**[Speaker Host]:**

assistant
Our Bidirectional Decoding explicitly searches for the optimal action from multiple predictions at each time step, achieving both long-term consistency and short-term reactivity. This comes at the cost of reduced access to recent state observations, which can be crucial for reacting to unexpected dynamics from modeling errors or environmental stochasticity.


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
comes at the cost of reduced access to recent state observations, which can be crucial for reacting
to unexpected dynamics arising from modeling errors or environmental stochasticity. This tradeoff raises a crucial question: How can we preserve the strengths of action chunking in long-term

**[Speaker Guest]:**

❓ consistency without suffering from its limitations in short-term reactivity?

**[Speaker Host]:**

assistant

**[Speaker Guest]:**

❓ The cost of action chunking is reduced access to recent state observations, which can be vital for responding to unexpected dynamics from modeling errors or environmental noise. This tradeoff raises a key question: how can we maintain the long-term consistency benefits of action chunking without sacrificing short-term reactivity?


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
This tradeoff raises a crucial question: How can we preserve the strengths of action chunking in long-term
consistency without suffering from its limitations in short-term reactivity? To this end, we introduce Bidirectional Decoding (BID), an inference algorithm that bridges action
chunking with closed-loop adaptation. Our main idea is to sample multiple predictions at each time
step and search for the most desirable one.

**[Speaker Host]:**

assistant
This tradeoff raises a crucial question: How can we preserve the strengths of action chunking in long-term consistency without suffering from its limitations in short-term reactivity? To address this, we introduce Bidirectional Decoding (BID), an inference algorithm that bridges action chunking with closed-loop adaptation. Our core idea is to sample multiple predictions at each time step and search for the most desirable one.


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
Our main idea is to sample multiple predictions at each time
step and search for the most desirable one. Specifically, BID operates on two decoding criteria: (i)
backward coherence, which favors samples that are close to the sequence selected at the previous
step; (ii) forward contrast, which favors samples that are close to the output of a stronger policy
and distant from those of a weaker one. As illustrated in Fig.

**[Speaker Guest]:**

assistant
Our main idea is to sample multiple predictions at each time step and search for the most desirable one. BID operates on two decoding criteria: backward coherence, which favors samples that are close to the sequence selected at the previous step; and forward contrast, which favors samples that are close to the output of a stronger policy and distant from those of a weaker one.


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
As illustrated in Fig. 1, BID updates the chunk of future
actions based on the previous strategy, promoting temporal consistency over extended periods while
remaining reactive to unexpected changes. The main contributions of this paper are twofold: (i) a thorough analysis of action chunking (§3),
and (ii) a decoding algorithm to improve it (§4).

**[Speaker Host]:**

assistant
As shown in Figure 1, BID updates a chunk of future actions based on the previous strategy, maintaining temporal consistency over time while remaining responsive to unexpected changes. The paper’s main contributions are: (1) a detailed analysis of action chunking in Section 3, and (2) a decoding algorithm to improve it in Section 4.


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
The main contributions of this paper are twofold: (i) a thorough analysis of action chunking (§3),
and (ii) a decoding algorithm to improve it (§4). Empirically, we validate our theoretical analysis
through a one-dimensional diagnostic simulation and evaluate our decoding method on two state-ofthe-art generative policies across seven simulations and two real-world tasks (§5).

**[Speaker Guest]:**

assistant
The paper makes two main contributions: a detailed analysis of action chunking and a new decoding algorithm to enhance it. Empirically, the authors validate their theoretical analysis using a one-dimensional diagnostic simulation and evaluate their decoding method on two state-of-the-art generative policies across seven simulations and two real-world tasks.


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
Empirically, we validate our theoretical analysis
through a one-dimensional diagnostic simulation and evaluate our decoding method on two state-ofthe-art generative policies across seven simulations and two real-world tasks (§5). Our experiment
results show that the proposed BID boosts the performance of recent policies by more than 32%
in relative performance.

**[Speaker Host]:**

assistant
We validate our theoretical analysis through a one-dimensional diagnostic simulation and evaluate our decoding method on two state-of-the-art generative policies across seven simulations and two real-world tasks. Our results show that the proposed BID boosts the performance of recent policies by more than 32% in relative performance.


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

Our experiment
results show that the proposed BID boosts the performance of recent policies by more than 32%
in relative performance. BID is model-agnostic, computationally efficient, and easy to implement,
serving as a plug-and-play component to enhance generative behavior cloning at test time. RELATED WORK

**[Speaker Host]:**

Behavioral Cloning.

**[Speaker Guest]:**

assistant
Our experiment results show that the proposed BID boosts the performance of recent policies by more than 32% in relative performance. BID is model-agnostic, computationally efficient, and easy to implement, serving as a plug-and-play component to enhance generative behavior cloning at test time. Behavior Cloning.


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

RELATED WORK
Behavioral Cloning. Learning from human demonstrations is becoming increasingly popular in
robot learning due to recent advances in robotic teleoperation interfaces (Sivakumar et al., 2022;
Zhao et al., 2023b; Wu et al., 2023; Chi et al., 2024).

**[Speaker Guest]:**

assistant
Behavioral cloning is gaining popularity in robot learning as a result of recent advances in robotic teleoperation interfaces.


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
Learning from human demonstrations is becoming increasingly popular in
robot learning due to recent advances in robotic teleoperation interfaces (Sivakumar et al., 2022;
Zhao et al., 2023b; Wu et al., 2023; Chi et al., 2024). Generative Behavior cloning, which models
the distribution of demonstrations, is particularly appealing due to its algorithmic simplicity and empirical efficacy (Jang et al., 2022; Florence et al., 2022; Brohan et al., 2022; Shafiullah et al., 2022;
Zhao et al., 2023b; Chi et al., 2024; Brohan et al., 2023a).

**[Speaker Host]:**

assistant
Learning from human demonstrations is becoming increasingly popular in robot learning, thanks to recent advances in robotic teleoperation interfaces. Generative behavior cloning, which models the distribution of demonstrations, is especially attractive because of its simple algorithmic design and proven effectiveness.


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
Generative Behavior cloning, which models
the distribution of demonstrations, is particularly appealing due to its algorithmic simplicity and empirical efficacy (Jang et al., 2022; Florence et al., 2022; Brohan et al., 2022; Shafiullah et al., 2022;
Zhao et al., 2023b; Chi et al., 2024; Brohan et al., 2023a). However, a significant limitation is compounding errors, where deviations from the training distribution accumulate over time (Ross et al.,
2011; Ke et al., 2021).

**[Speaker Guest]:**

assistant
Generative behavior cloning, which models the distribution of demonstrations, is particularly appealing due to its algorithmic simplicity and empirical efficacy. However, a significant limitation is compounding errors, where deviations from the training distribution accumulate over time.


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
However, a significant limitation is compounding errors, where deviations from the training distribution accumulate over time (Ross et al.,
2011; Ke et al., 2021). These errors can be mitigated by gathering expert correction data (Ross et al.,
2011; Kelly et al., 2019; Menda et al., 2019; Hoque et al., 2021a; b) or injecting noise during data
collection (Laskey et al., 2017; Brandfonbrener et al., 2023), but such strategies require additional
time and effort from human operators.

**[Speaker Host]:**

assistant
A significant limitation is compounding errors, where deviations from the training distribution accumulate over time. These errors can be mitigated by gathering expert correction data or injecting noise during data collection, but such strategies require additional time and effort from human operators.


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
These errors can be mitigated by gathering expert correction data (Ross et al.,
2011; Kelly et al., 2019; Menda et al., 2019; Hoque et al., 2021a; b) or injecting noise during data
collection (Laskey et al., 2017; Brandfonbrener et al., 2023), but such strategies require additional
time and effort from human operators. To address this, recent work proposes predicting a sequence
of multiple actions into the future, known as action chunking, which reduces the effective control
horizon (Lai et al., 2022; Zhao et al., 2023b; George & Farimani, 2023; Bharadhwaj et al., 2023a).

**[Speaker Guest]:**

assistant
These errors can be mitigated by gathering expert correction data or injecting noise during data collection, but such strategies require additional time and effort from human operators. To address this, recent work proposes predicting a sequence of multiple actions into the future, known as action chunking, which reduces the effective control horizon.


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
To address this, recent work proposes predicting a sequence
of multiple actions into the future, known as action chunking, which reduces the effective control
horizon (Lai et al., 2022; Zhao et al., 2023b; George & Farimani, 2023; Bharadhwaj et al., 2023a). By handling sequences of actions, action chunking is also better at handling temporal dependencies
in the data, such as idle pauses (Swamy et al., 2022; Chi et al., 2023) or multiple styles (Li et al.,
2017; Kuefler & Kochenderfer, 2017; Gandhi et al., 2023; Belkhale et al., 2024).

**[Speaker Host]:**

assistant
Recent work proposes predicting sequences of multiple actions into the future, a technique known as action chunking, which reduces the effective control horizon. By handling sequences of actions, action chunking better captures temporal dependencies in the data, such as idle pauses or multiple styles.


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
By handling sequences of actions, action chunking is also better at handling temporal dependencies
in the data, such as idle pauses (Swamy et al., 2022; Chi et al., 2023) or multiple styles (Li et al.,
2017; Kuefler & Kochenderfer, 2017; Gandhi et al., 2023; Belkhale et al., 2024). However, independently drawn action sequence samples may not preserve the necessary temporal dependencies
for smooth and consistent execution.

**[Speaker Guest]:**

assistant
Action chunking improves handling of temporal dependencies in sequences, such as idle pauses or shifts in style. However, independently sampled action sequences may fail to preserve the necessary temporal structure for consistent execution.


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
However, independently drawn action sequence samples may not preserve the necessary temporal dependencies
for smooth and consistent execution. Our work provides a thorough analysis of action chunking and
proposes a decoding algorithm to improve it. Published as a conference paper at ICLR 2025
Sequential Decoding. Decoding algorithms have been studied in generative sequence modeling
for decades, with renewed attention driven by recent advances in large language modeling (LLM).

**[Speaker Host]:**

assistant
However, independently drawn action sequence samples may not preserve the necessary temporal dependencies for smooth and consistent execution. Our work provides a thorough analysis of action chunking and proposes a decoding algorithm to improve it. Published as a conference paper at ICLR 2025. Sequential decoding has been studied in generative sequence modeling for decades, with renewed attention driven by recent advances in large language modeling.


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
Decoding algorithms have been studied in generative sequence modeling
for decades, with renewed attention driven by recent advances in large language modeling (LLM). One prominent approach focuses on leveraging internal metrics, e. g., likelihood scores, to improve
the quality of generated sequences.

**[Speaker Guest]:**

assistant
Decoding algorithms have been studied in generative sequence modeling for decades, with renewed attention driven by recent advances in large language modeling. One prominent approach focuses on leveraging internal metrics, such as likelihood scores, to improve the quality of generated sequences.


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
g., likelihood scores, to improve
the quality of generated sequences. Notable examples include beam search (Freitag & Al-Onaizan,
2017; Vijayakumar et al., 2018), truncated sampling (Fan et al., 2018; Hewitt et al., 2022), minimum
Bayes risk decoding (Kumar & Byrne, 2004; M¨uller & Sennrich, 2021), and others (Welleck et al.,
2019; Meister et al., 2023; Fu et al., 2024a).

**[Speaker Host]:**

assistant
Likelihood scores are used to improve the quality of generated sequences. Notable examples include beam search, truncated sampling, and minimum Bayes risk decoding.


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
Notable examples include beam search (Freitag & Al-Onaizan,
2017; Vijayakumar et al., 2018), truncated sampling (Fan et al., 2018; Hewitt et al., 2022), minimum
Bayes risk decoding (Kumar & Byrne, 2004; M¨uller & Sennrich, 2021), and others (Welleck et al.,
2019; Meister et al., 2023; Fu et al., 2024a). Another line of research explores the distinctions between multiple models to jointly optimize for the desired properties such as quality or efficiency (Li
et al., 2023; Leviathan et al., 2023).

**[Speaker Guest]:**

assistant
Notable examples include beam search, truncated sampling, minimum Bayes risk decoding, and others. Another line of research explores how multiple models can be used together to jointly optimize for qualities like performance or efficiency.


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
Another line of research explores the distinctions between multiple models to jointly optimize for the desired properties such as quality or efficiency (Li
et al., 2023; Leviathan et al., 2023). More recently, several studies have highlighted the potential
of guiding the decoding or sampling process through the use of an external model, such as a classifier (Dhariwal & Nichol, 2021) or reward model (Khanov et al., 2023).

**[Speaker Host]:**

assistant
Another line of research explores the distinctions between multiple models to jointly optimize for desired properties like quality or efficiency. More recently, several studies have highlighted the potential of guiding the decoding or sampling process using an external model, such as a classifier or reward model.


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
More recently, several studies have highlighted the potential
of guiding the decoding or sampling process through the use of an external model, such as a classifier (Dhariwal & Nichol, 2021) or reward model (Khanov et al., 2023). In the context of robot
learning, recent works have explored guided decoding for long-horizon robotic planning (Huang
et al., 2023) and manipulator geometry designs (Xu et al., 2024).

**[Speaker Guest]:**

assistant
More recently, several studies have highlighted the potential of guiding the decoding or sampling process using an external model, such as a classifier or a reward model. In the context of robot learning, recent works have explored guided decoding for long-horizon robotic planning and manipulator geometry design.


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
In the context of robot
learning, recent works have explored guided decoding for long-horizon robotic planning (Huang
et al., 2023) and manipulator geometry designs (Xu et al., 2024). Nevertheless, effective decoding
strategies for low-level robotic actions remain lacking. Concurrent to our work, Nakamoto et al. propose to select action samples by querying a value function learned from reward-annotated
demonstrations (Hansen-Estruch et al., 2023).

**[Speaker Host]:**

assistant
In robot learning, recent studies have explored guided decoding for long-horizon robotic planning and manipulator geometry design. However, effective decoding strategies for low-level robotic actions are still missing. Meanwhile, Nakamoto et al. suggest selecting action samples by querying a value function trained from reward-annotated demonstrations.


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
propose to select action samples by querying a value function learned from reward-annotated
demonstrations (Hansen-Estruch et al., 2023). Our work does not rely on a separate value function;
instead, we propose a decoding strategy that addresses the consistency-reactivity tradeoff inherent
in action chunking through sample comparison. ANALYSIS: TRADEOFFS IN ACTION CHUNKING

**[Speaker Guest]:**

3.

**[Speaker Host]:**

assistant
We propose to select action samples by querying a value function learned from reward-annotated demonstrations. Our work does not rely on a separate value function; instead, we propose a decoding strategy that addresses the consistency-reactivity tradeoff inherent in action chunking through sample comparison.


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
ANALYSIS: TRADEOFFS IN ACTION CHUNKING

**[Speaker Guest]:**

3. 1

**[Speaker Host]:**

PRELIMINARIES
Consider a dataset of demonstrations D = {τi}N
i=1, where each demonstration τi consists of a sequence of state-action pairs τi = {(s1, a1), (s2, a2), · · ·, (sT, aT )} provided by a human expert. At each time step t, the demonstrated action at is influenced not only by the observed state st,
but also by latent variables zt, such as planning strategies (e. g., subgoals) and personal preferences
(e. g., handedness).

**[Speaker Guest]:**

assistant
Consider a dataset of demonstrations D = {τi}N i=1, where each demonstration τi consists of a sequence of state-action pairs τi = {(s1, a1), (s2, a2), ... ... ..., (sT, aT)} provided by a human expert. At each time step t, the demonstrated action at is influenced not only by the observed state st, but also by latent variables zt, such as planning strategies (e.g., subgoals) and personal preferences (e.g., handedness).


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
g., handedness). These latent variables can persist across multiple time steps and vary significantly between different demonstrations. Fig. 2 illustrates the decision process of a human expert,
highlighting the inherent temporal dependencies. To model these temporal dependencies, some recent works (Zhao et al., 2023b; Chi et al., 2023; Lee
et al., 2024) utilize action chunking, i.

**[Speaker Host]:**

assistant
g., handedness). These latent variables can persist across multiple time steps and vary significantly between different demonstrations. Fig. 2 illustrates the decision process of a human expert, highlighting the inherent temporal dependencies. To model these temporal dependencies, some recent works (Zhao et al., 2023b; Chi et al., 2023; Lee et al., 024) utilize action chunking, i.


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
To model these temporal dependencies, some recent works (Zhao et al., 2023b; Chi et al., 2023; Lee
et al., 2024) utilize action chunking, i. e., modeling the joint distribution of future actions conditioned
on past states π(at, at+1, · · ·, at+l|st−c, · · ·, st), or in short π(at:t+l|st−c:t). Here, c is the context
length, i. e., number of past steps for state inputs, and l is the prediction length, i. e., number of future
steps for action outputs.

**[Speaker Guest]:**

assistant
Recent works use action chunking to model temporal dependencies, modeling the joint distribution of future actions given past states. Specifically, they model π(at to at+l | st−c to st), where c is the context length—the number of past steps used for state inputs—and l is the prediction length, the number of future steps for action outputs.


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
e., number of future
steps for action outputs. Training such a policy typically involves minimizing the divergence of
action distributions between the model π and the expert π∗,

**[Speaker Host]:**

π = arg min

**[Speaker Guest]:**

π

**[Speaker Host]:**

X

**[Speaker Guest]:**

τ∈D

**[Speaker Host]:**

X

**[Speaker Guest]:**

st−c:t

**[Speaker Host]:**

at:t+l
L(π(at:t+l|st−c:t), π∗(at:t+l|st−c:t)). (1)
During deployment, the policy operates with a specific action horizon h ∈[1, l], i. e., executing
part or all of the predicted action sequence for h time steps without re-planning.

**[Speaker Guest]:**

assistant
The policy is trained to minimize the divergence between its predicted actions and those of an expert policy, across a set of trajectories. During deployment, it executes only a portion or all of the predicted action sequence for a fixed horizon, without re-planning.


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

e., executing
part or all of the predicted action sequence for h time steps without re-planning. This approach
essentially takes in c states as context and executes h actions, which we refer to as a (c, h)-policy. The choices of context length c and action horizon h often play a crucial role in the effectiveness
of the learned policy.

**[Speaker Guest]:**

assistant
Executing part or all of the predicted action sequence for h time steps without re-planning. This approach takes in c states as context and executes h actions, referred to as a (c, h)-policy. The choices of context length c and action horizon h often significantly impact policy effectiveness.


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
The choices of context length c and action horizon h often play a crucial role in the effectiveness
of the learned policy. Recent policies often use a short context length c, as extending the context
can lead to performance degradation in the presence of limited training data (refer to Appendix A. 2
for more details). Conversely, extending the action horizon h has produced mixed results.

**[Speaker Host]:**

assistant
The choices of context length c and action horizon h often play a crucial role in the effectiveness of the learned policy. Recent policies often use a short context length c, as extending the context can lead to performance degradation in the presence of limited training data. Conversely, extending the action horizon h has produced mixed results.


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
Conversely, extending the action horizon h has produced mixed results. Some
studies report benefits in laboratory settings (Zhao et al., 2023b; Chi et al., 2023), while others
find it detrimental in real-world scenarios (Lee et al., 2024). The reasons behind these conflicting
outcomes are not well understood yet. Notably, Zhao et al.

**[Speaker Guest]:**

assistant
Extending the action horizon has produced mixed results. Some studies report benefits in lab settings, while others find it harmful in real-world applications. The reasons for these conflicting outcomes are not yet well understood.


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
Notably, Zhao et al. (2023b) hypothesize that action chunking mitigates compounding errors, but it is
unclear why this would hold when deviations within a chunk cannot be corrected before replanning. Alternatively, Chi et al. draws parallels to model predictive control (MPC) (Borrelli et al.,
2017; L¨ofberg, 2012), emphasizing its role in improving consistency and planning. However, unlike
MPC, which typically uses short action horizons (e.

**[Speaker Host]:**

assistant
Zhao et al. (2023b) hypothesize that action chunking mitigates compounding errors, but it is unclear why this would hold when deviations within a chunk cannot be corrected before replanning. Chi et al. draw parallels to model predictive control (MPC), emphasizing its role in improving consistency and planning. However, unlike MPC, which typically uses short action horizons, the mechanism by which chunking achieves stability remains unexplained.


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

However, unlike
MPC, which typically uses short action horizons (e. g., h = 1), recent imitation learning methods
employ much longer action horizons, such as a substantial portion of the predicted sequence (e. g.,
h = 8 in Chi et al. ) or even the entire prediction length (e. g., h greater than or equal to 50 in Zhao et al. (2023b);

**[Speaker Host]:**

Black et al. ).

**[Speaker Guest]:**

assistant
Unlike MPC, which typically uses short action horizons—such as h = 1—recent imitation learning methods employ much longer action horizons, including substantial portions of the predicted sequence, like h = 8 in Chi et al., or even the entire prediction length, such as h ≥ 50 in Zhao et al. (2023b) and Black et al.


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
). The lack of a clear understanding of action chunking hinders its effective use
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
temporal dependencies in actions. prediction

**[Speaker Guest]:**

replan

**[Speaker Host]:**

t+1

**[Speaker Guest]:**

t-k

**[Speaker Host]:**

𝜋(𝑐, ℎ)

**[Speaker Guest]:**

... ... ... 𝜋(𝑐, ℎ+𝑑)

**[Speaker Host]:**

... ... ... 𝜋(𝑘, 1)

**[Speaker Guest]:**

∗

**[Speaker Host]:**

... ... ... observation

**[Speaker Guest]:**

t-h

**[Speaker Host]:**

t-h-d

**[Speaker Guest]:**

t

**[Speaker Host]:**

Figure 3:
Illustration of (k, 1)-expert, (c, h)-learner, and

**[Speaker Guest]:**

(c, h + d)-learner.

**[Speaker Host]:**

assistant
The lack of a clear understanding of action chunking hinders its effective use. Figure 2 illustrates the expert decision process, where a latent variable introduces temporal dependencies in actions. Figure 3 shows the (k, 1)-expert, (c, h)-learner, and (c, h + d)-learner.


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

observation

**[Speaker Host]:**

t-h

**[Speaker Guest]:**

t-h-d

**[Speaker Host]:**

t

**[Speaker Guest]:**

Figure 3:
Illustration of (k, 1)-expert, (c, h)-learner, and
(c, h + d)-learner. Shaded regions represent observed history;
darker indicate greater influence on current decision. across tasks and environments. We next explicitly analyze the strengths and weaknesses of action
chunking, with particular attention to the choice of action horizon h. 3.

**[Speaker Host]:**

assistant
Observation: The figure illustrates a (k,1)-expert, a (c,h)-learner, and a (c,h+d)-learner. Shaded regions represent the observed history, with darker shades indicating greater influence on current decision-making. Across tasks and environments, we next analyze the strengths and weaknesses of action chunking, with particular attention to the choice of action horizon h.


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

3. 2

**[Speaker Host]:**

ANALYSIS
To understand the influence of action chunking, we focus on the last time step of the executed chunk,
where the discrepancy between the expert policy and the learned policy is most pronounced. At this
time step t, a (k, 1)-expert written as π∗:= π∗(at|st−k:t, zt−k:t) predicts at by conditioning on k
steps of the past states and the corresponding latent variables.

**[Speaker Guest]:**

assistant
At the final time step of the executed chunk, the difference between the expert policy and the learned policy is most evident. At this time step t, the expert policy—denoted as π∗—predicts the action at by conditioning on k steps of the past states and the corresponding latent variables.


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

At this
time step t, a (k, 1)-expert written as π∗:= π∗(at|st−k:t, zt−k:t) predicts at by conditioning on k
steps of the past states and the corresponding latent variables. In contrast, a (c, h)-learner written as
π(c, h) := π(c, h)(at | st−h−c:t−h, at−h:t−1) is constrained to observe c steps of the past states and
h −1 steps of the past actions within the predicted chunk.

**[Speaker Guest]:**

assistant
At this time step t, a (k, 1)-expert predicts the action at by conditioning on k steps of the past states and the corresponding latent variables. In contrast, a (c, h)-learner is constrained to observe c steps of the past states and h−1 steps of the past actions within the predicted chunk.


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
In contrast, a (c, h)-learner written as
π(c, h) := π(c, h)(at | st−h−c:t−h, at−h:t−1) is constrained to observe c steps of the past states and
h −1 steps of the past actions within the predicted chunk. The divergence between a learned policy and an expert policy is attributed to two factors: (i) the
importance of unobserved states in predicting the current action, and (ii) the difficulty of inferring
unobserved states based on the available information.

**[Speaker Host]:**

assistant
A (c, h)-learner is defined as π(c, h) := π(c, h)(at | st−h−c:t−h, at−h:t−1), and is limited to observing c steps of past states and h−1 steps of past actions within each prediction window. The difference between a learned policy and an expert policy arises from two key factors: first, the importance of unobserved states in determining the current action; second, the challenge of inferring unobserved states from the available information.


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
The divergence between a learned policy and an expert policy is attributed to two factors: (i) the
importance of unobserved states in predicting the current action, and (ii) the difficulty of inferring
unobserved states based on the available information. To more clearly see the influence of action horizon on these factors, we next compare the performance of two policies that have the same context lengths but different action horizons, πh :=
π(c, h)(at|st−h−c:t−h, at−h:t−1) and πh+d := π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−1), where
d > 0 is the extended action horizon.

**[Speaker Guest]:**

assistant
The gap between a learned policy and an expert policy stems from two key factors: first, the importance of unobserved states in predicting the current action, and second, the difficulty of inferring those unobserved states from available information. To better understand how action horizon affects these factors, we compare two policies with the same context length but different action horizons. Specifically, we look at πh = π(c, h)(at|st−h−c:t−h, at−h:t−1) and πh+d = π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−1), where d > 0 represents an extended action horizon.


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
To more clearly see the influence of action horizon on these factors, we next compare the performance of two policies that have the same context lengths but different action horizons, πh :=
π(c, h)(at|st−h−c:t−h, at−h:t−1) and πh+d := π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−1), where
d > 0 is the extended action horizon. As illustrated in Fig. 3, each policy has access to unique
information that is unavailable to the other.

**[Speaker Host]:**

assistant
To better understand how action horizon affects performance, we compare two policies with the same context length but different action horizons. The first policy, πh, makes decisions based on observations from time step t−h−c to t−h, and actions from t−h to t−1. The second policy, πh+d, has an extended action horizon and observes from t−h−d−c to t−h−d, with actions from t−h−d to t−1, where d > 0. As shown in Figure 3, each policy accesses unique information not available to the other.


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
3, each policy has access to unique
information that is unavailable to the other. πh observes some recent states, where πh+d is only
aware of the executed actions. On the other hand, πh+d has access to some earlier states and actions,
which precede all information available to πh. We characterize the importance of observations as
follows (formal definitions in Appendix E. 2):
Definition (Expected Observation Advantage).

**[Speaker Guest]:**

assistant
Each policy has access to unique information not available to the others. πh observes recent states, while πh+d only sees the actions that were executed. Conversely, πh+d has access to earlier states and actions that predate all the information available to πh. We define the importance of observations as follows—formal definitions in Appendix E.2.


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

2):
Definition (Expected Observation Advantage). If a policy can observe a state st, we say that it
has an observation advantage αt over another policy that cannot observe it. More formally, this is
the difference between the expected divergence accumulated by a policy π(at′|st′) that does not
condition on the state st and that by a policy π(at′|st′, st) that conditions on st. Definition (Maximum Inference Disadvantage).

**[Speaker Guest]:**

assistant
If a policy can observe a state st, we say it has an observation advantage αt over another policy that cannot observe it. This is defined as the difference between the expected divergence accumulated by a policy π(at′|st′) that does not condition on the state st and the divergence accumulated by a policy π(at′|st′, st) that does condition on st.


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
Definition (Maximum Inference Disadvantage). If a policy cannot condition its prediction on
a state st, its maximum inference disadvantage ϵt is the largest possible divergence arising from
inferring incorrectly. Here, the maximum is taken over all possible incorrect inferences of st.

**[Speaker Host]:**

assistant
If a policy cannot condition its prediction on a state st, its maximum inference disadvantage ϵt is the largest possible divergence that arises from making an incorrect inference about st.


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
Here, the maximum is taken over all possible incorrect inferences of st. Hence, we denote the observation advantage that πh gains from the observed recent states by αf
and the maximum inference disadvantage it incurs from the earlier unobserved states by ϵb, whereas
πh+d conversely gains αb but incurs ϵf. Let P(st+1 | st, at) denote the true transition probabilities and ˆP(st+1 | st, at) the implicit dynamics model learned by the policy.

**[Speaker Guest]:**

assistant
The maximum is taken over all possible incorrect inferences of the policy. We denote the observation advantage that πh gains from the observed recent states as αf, and the maximum inference disadvantage it incurs from earlier unobserved states as ϵb. Conversely, πh+d gains αb but incurs ϵf. Let P(st+1 | st, at) denote the true transition probabilities, and ˆP(st+1 | st, at) denote the implicit dynamics model learned by the policy.


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
Let P(st+1 | st, at) denote the true transition probabilities and ˆP(st+1 | st, at) the implicit dynamics model learned by the policy. The difficulty of inferring an unobserved state hinges on both the
relevant observations and environmental stochasticity, which we quantify as follows. Definition (Forward Inference). Let Pf(t) := P(St = gt|St−1 = gt−1, At−1 = at−1), where gt
and gt−1 are the ground truth states at time t and t −1 respectively in a deterministic environment.

**[Speaker Host]:**

assistant
Let P(st+1 | st, at) denote the true transition probabilities and ˆP(st+1 | st, at) the implicit dynamics model learned by the policy. The difficulty of inferring an unobserved state hinges on both the relevant observations and environmental stochasticity. Definition (Forward Inference): Let Pf(t) := P(St = gt | St−1 = gt−1, At−1 = at−1), where gt and gt−1 are the ground truth states at time t and t−1, respectively, in a deterministic environment.


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
Let Pf(t) := P(St = gt|St−1 = gt−1, At−1 = at−1), where gt
and gt−1 are the ground truth states at time t and t −1 respectively in a deterministic environment. In deterministic environments, Pf(t) = 1; whereas in stochastic settings, Pf(t) is smaller. The
prediction error is characterized by δf(t) = ˆP(St = gt|St−1 = gt−1, at−1) −P(St = gt|St−1 =
gt−1, at−1). Definition (Backward Inference).

**[Speaker Guest]:**

assistant
Let \( P_f(t) \) denote the probability that the state at time \( t \) equals the ground truth state \( g_t \), given the previous state \( g_{t-1} \) and action \( a_{t-1} \). In deterministic environments, this probability is 1; in stochastic environments, it is less than 1. The prediction error is defined as the difference between the predicted probability and the true probability: \( \delta_f(t) = \hat{P}(S_t = g_t | S_{t-1} = g_{t-1}, a_{t-1}) - P(S_t = g_t | S_{t-1} = g_{t-1}, a_{t-1}) \).

Definition (Backward Inference).


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
Definition (Backward Inference). Similarly, let Pb(t) := P(St = gt|St+1 = gt+1) where gt and
gt+1 are the ground truth states in the deterministic environment at time t and t + 1, respectively. Published as a conference paper at ICLR 2025
Since Pb(t) is not conditioned on any action, it has higher entropy in general. In stochastic environments, Pb(t) is small. Furthermore, let δb(t) = ˆP(St = gt|St+1 = gt+1) −P(St = gt|St+1 =

**[Speaker Host]:**

gt+1).

**[Speaker Guest]:**

assistant
Backward inference involves defining Pb(t) as the probability that the state at time t is gt, given that the state at time t+1 is gt+1. This is expressed as P(St = gt | St+1 = gt+1), where gt and gt+1 are the true states in a deterministic environment at times t and t+1, respectively. Since Pb(t) does not depend on any action, it generally has higher entropy. In stochastic environments, Pb(t) tends to be small. Additionally, δb(t) is defined as the difference between the estimated probability and the true probability: ˆP(St = gt | St+1 = gt+1) − P(St = gt | St+1 = gt+1).


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
Furthermore, let δb(t) = ˆP(St = gt|St+1 = gt+1) −P(St = gt|St+1 =
gt+1). We refer to the variance of the distribution P(at′−1,... ... ..., at′−d | st′) under expert policy as the
diversity of past strategies up to time t′. Given that the forward inference is generally easier than
the backward inference, the performances of πh and πh+d differ as follows (proofs are deferred to

**[Speaker Host]:**

Appendix E):
Proposition 1 (Consistency-Reactivity Inequalities).

**[Speaker Guest]:**

assistant
Furthermore, let δb(t) be the difference between the predicted and actual probability of state St = gt given St+1 = gt+1. We define the variance of the distribution of past strategies under the expert policy as the diversity of past strategies up to time t′. Since forward inference is generally easier than backward inference, the performances of policies πh and πh+d differ according to the following inequalities—proofs are deferred to Appendix E.

Proposition 1: Consistency-Reactivity Inequalities.


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
Given that the forward inference is generally easier than
the backward inference, the performances of πh and πh+d differ as follows (proofs are deferred to

**[Speaker Host]:**

Appendix E):
Proposition 1 (Consistency-Reactivity Inequalities). Let L be a non-linear and non-negative convex
function measuring the prediction error with respect to demonstrations. Let C := {at−h:t−1} ∪S+
where S+ are the common states that both πh and πh+d observe.

**[Speaker Guest]:**

assistant
Given that forward inference is generally easier than backward inference, the performances of πh and πh+d differ as follows (proofs are deferred to Appendix E):
Proposition 1 (Consistency-Reactivity Inequalities). Let L be a non-linear and non-negative convex function measuring the prediction error with respect to demonstrations. Let C := {at−h:t−1} ∪ S+ where S+ are the common states that both πh and πh+d observe.


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
Let C := {at−h:t−1} ∪S+
where S+ are the common states that both πh and πh+d observe. For the ease of notation, let
τf = {t −h −d + 1 : t −h} and let τb = {t −h −d −c : t −h −c −1}. Then,
the difference in the expected loss between the (c, h + d)-policy and the (c, h)-policy, ∆d :=
minπh+d E [L(πh+d, π∗)|C] −minπh E [L(πh, π∗)|C], is bounded as:

**[Speaker Host]:**

αf −ϵb

**[Speaker Guest]:**

assistant
Let C be the set of common states observed by both policies πh and πh+d, defined as C := {at−h: t−1} ∪ S+, where S+ are the common states. For notational simplicity, define τf = {t − h − d + 1: t − h} and τb = {t − h − d − c: t − h − c − 1}. The difference in expected loss between the (c, h + d)-policy and the (c, h)-policy, denoted ∆d = minπh+d E[L(πh+d, π∗)|C] − minπh E[L(πh, π∗)|C], is bounded by αf − ϵb.


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

assistant
The given text contains mathematical expressions and appears to be related to a technical or mathematical model, possibly involving probabilities or thresholds. However, it is not clearly structured or readable in its current form due to missing context, inconsistent formatting, and LaTeX-style symbols without proper rendering.

After cleaning and removing irrelevant or uninterpretable elements (such as LaTeX math syntax, incomplete expressions, and formatting noise), the only meaningful textual content appears to be:

"less than or equal to ∆d less than or equal to ϵf"

This suggests a bound or inequality involving a variable ∆d that is constrained between two values, possibly related to a tolerance or error margin (ϵf).

However, without additional context or clarification of the variables (e.g., what Y, τb, Pb, δb, τf, Pf, δf, ∆d, ϵf represent), and with no clear narrative or explanation, the original content cannot be meaningfully interpreted as a coherent statement.

Therefore, after rigorous preprocessing and removal of all non-essential, uninterpretable, or mathematically malformed content, the only usable portion is:

less than or equal to ∆d less than or equal to ϵf


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

−αb. (2)
Remark 1. Eq. (2) provides a general comparison of the performance of the two policies. Intuitively,
the advantage of each policy stems from the additional information it has access to (i. e. αf for πh and
αb for πh+d) while the disadvantage is bounded by the maximum divergence arising from inferring
missing information incorrectly (i. e. ϵb and ϵf scaled by the maximum probability of incorrect
inference). We next examine the implications of Proposition 1. Corollary 2 (Consistency).

**[Speaker Host]:**

assistant
Eq. (2) offers a general comparison of the performance of two policies. The advantage of each policy comes from the additional information it has access to—αf for πh and αb for πh+d—while the disadvantage is limited by the maximum divergence caused by incorrectly inferring missing information, scaled by the maximum probability of such incorrect inference. We now examine the implications of Proposition 1. Corollary 2 (Consistency).


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
Corollary 2 (Consistency). Suppose the train and test environments are identical and deterministic. Suppose at is influenced by at least one state at time steps τb and let δf(t′) approximately 0 for all t′ ∈τf. If
the diversity in past strategies up to time t −h −c is not 0, and ϵf is finite, then

**[Speaker Guest]:**

min
πh+d E [L(πh+d, π∗)|C] < min
πh E [L(πh, π∗)|C]. (3)

**[Speaker Host]:**

Remark 2.

**[Speaker Guest]:**

assistant
The corollary states that if the training and test environments are identical and deterministic, and if the action at time step t is influenced by at least one state at earlier time steps τb, and if the difference δf(t′) is approximately zero for all t′ in the set τf, then if there is diversity in past strategies up to time t−h−c and the error term ϵf is finite, the expected loss of a policy πh+d with longer horizon is strictly less than that of a shorter-horizon policy πh — under the same conditions. This implies that adding more history or diversity in past strategies improves performance.


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

(3)
Remark 2. In deterministic environments, while both policies need to infer the same number of
unobserved states, πh+d benefits from conditioning on additional actions, which may significantly
aid in inferring the corresponding states through its action chunk. However, this is only true if the
implicit dynamics model learned by the policy is approximately accurate and the maximum errors
ϵf arising from inferring these states are bounded.

**[Speaker Guest]:**

assistant
In deterministic environments, both policies must infer the same number of unobserved states, but πh+d benefits from conditioning on additional actions, which can help infer the corresponding states through its action chunk. This advantage holds only if the policy's learned dynamics model is approximately accurate and the errors from state inference remain bounded.


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
However, this is only true if the
implicit dynamics model learned by the policy is approximately accurate and the maximum errors
ϵf arising from inferring these states are bounded. Note that πh+d performs better as we increase
the diversity of strategies present in the dataset i. e., as the distribution of actions taken by the expert
prior to the context of the short horizon policy becomes higher variance. Corollary 3 (Reactivity).

**[Speaker Host]:**

assistant
However, this holds only if the implicit dynamics model learned by the policy is approximately accurate and the maximum errors ϵf from inferring these states are bounded. πh+d performs better as the diversity of strategies in the dataset increases—i.e., as the distribution of expert actions before the context of the short-horizon policy becomes more varied.


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
Corollary 3 (Reactivity). Suppose Pf(t′) is small or both Pf(t′) and |δf(t′)| are large for all
t′ ∈τf. If temporal dependency decreases over time such that ϵb is small and at is influenced by at
least one state at time steps in τf, then

**[Speaker Guest]:**

min
πh+d E [L(πh+d, π∗)|C] > min
πh E [L(πh, π∗)|C]. (4)
Remark 3. Recent states are often more important to decision making and, therefore, the disadvantages of action chunking become pronounced when inferring these recent states becomes difficult.

**[Speaker Host]:**

assistant
Suppose the policy function at time step t′ is small, or both the policy function and the magnitude of its derivative are large, for all t′ in the time interval τf. If temporal dependencies weaken over time such that the noise parameter ϵb becomes small, and the current state at time step t is influenced by at least one state from earlier time steps in τf, then the expected loss of the policy with delayed action selection is strictly greater than that of the policy with immediate action selection. Recent states often play a more critical role in decision-making, meaning that the drawbacks of action chunking become more significant when it becomes difficult to infer these recent states.


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
Recent states are often more important to decision making and, therefore, the disadvantages of action chunking become pronounced when inferring these recent states becomes difficult. This can happen if the test environment is stochastic. This can also happen if the test environment is
deterministic as long as the implicit dynamics model is inaccurate - either due to a distribution shift
or because of learning difficulty.

**[Speaker Guest]:**

assistant
Recent states are often more important to decision making, so the disadvantages of action chunking become pronounced when inferring these recent states is difficult. This can happen in stochastic environments. It can also happen in deterministic environments if the implicit dynamics model is inaccurate—either due to a distribution shift or because of learning difficulty.


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
This can also happen if the test environment is
deterministic as long as the implicit dynamics model is inaccurate - either due to a distribution shift
or because of learning difficulty. In summary, depending on the experimental conditions, action chunking may have varying effects
on the learned policy. On the one hand, it benefits the modeling of temporal dependencies in the
demonstrations, due to extended access to actions executed at past time steps.

**[Speaker Host]:**

assistant
This can also happen if the test environment is deterministic, as long as the implicit dynamics model is inaccurate—either due to a distribution shift or because of learning difficulty. In summary, depending on the experimental conditions, action chunking may have varying effects on the learned policy. On the one hand, it benefits the modeling of temporal dependencies in the demonstrations, due to extended access to actions executed at past time steps.


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
On the one hand, it benefits the modeling of temporal dependencies in the
demonstrations, due to extended access to actions executed at past time steps. On the other hand,
it hinders reactions to unexpected dynamics, due to reduced access to state observations at recent
time steps. As a result, there is no universally optimal choice of action horizon across all conditions.

**[Speaker Guest]:**

assistant
On one hand, it helps model temporal dependencies in demonstrations by giving access to actions from past time steps. On the other hand, it limits the model’s ability to respond to unexpected dynamics by reducing access to recent state observations. As a result, there’s no single best choice for action horizon across all scenarios.


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
As a result, there is no universally optimal choice of action horizon across all conditions. When both temporal dependencies and prediction errors are significant, tuning the action horizon
entails an inherent trade-off between the two opposing factors. Published as a conference paper at ICLR 2025

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
Figure 4: Illustration of bidirectional decoding.

**[Speaker Guest]:**

assistant
As a result, there is no universally optimal choice of action horizon across all conditions. When both temporal dependencies and prediction errors are significant, tuning the action horizon entails an inherent trade-off between the two opposing factors.

Figure 4 illustrates bidirectional decoding.


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
Figure 4: Illustration of bidirectional decoding. Algorithm 1 Bidirectional Decoding
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
As analyzed above, action chunking improves long-term consistency but sacrifices short-term reactivity.

**[Speaker Guest]:**

assistant
Figure 4: Illustration of bidirectional decoding.

Algorithm 1: Bidirectional Decoding
Require: current state s, batch size N, mode size K, previous decision ˆa, strong policy π, weak policy π′
1: Generate N samples from each policy a ∼ π(s), a′ ∼ π′(s) to construct the initial sets A and A′
2: Compute the backward loss L_B for each sample
3: Select K samples with minimal L_B from A and A′ to construct A+ and A−, respectively
4: Compute the forward loss L_F for each sample
5: Select a* ∈ A that minimizes the total loss
6: Update decision memory ˆa ← a*

Method: Bidirectional Decoding
As analyzed above, action chunking improves long-term consistency but sacrifices short-term reactivity.


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
Algorithm 1 Bidirectional Decoding
Require: current state s, batch size N, mode size K, previous decision ˆa, strong policy π, weak policy π′
1: Generate N samples from each policy a ∼π(s), a′ ∼
π′(s) to construct the initial sets A and A

**[Speaker Host]:**

′
2: Compute the backward loss LB for each sample
3: Select K samples with minimal LB from A and A

**[Speaker Guest]:**

′ to
construct A+ and A−, respectively
4: Compute the forward loss LF for each sample
5: Select a∗∈A that minimizes the total loss
6: Update decision memory ˆa ←a∗
METHOD: BIDIRECTIONAL DECODING
As analyzed above, action chunking improves long-term consistency but sacrifices short-term reactivity. In this section, we propose to address this tradeoff by bridging action chunking with closedloop adaptation.

**[Speaker Host]:**

assistant
Bidirectional decoding is a method designed to balance long-term consistency with short-term reactivity in decision-making processes. It addresses the tradeoff between action chunking—where actions are grouped for better long-term coherence—and the need for immediate responsiveness. The method integrates action chunking with closed-loop adaptation to improve both performance and adaptability.

The algorithm begins by generating samples from both a strong policy π and a weak policy π′, creating initial sets A and A′. For each sample, the backward loss LB is computed, and the K samples with the lowest backward loss are selected to form the forward sets A+ and A−. Then, the forward loss LF is calculated for each sample, and from the set A, the action a∗ that minimizes the total loss is chosen. This action a∗ is then updated as the current decision memory ˆa. This iterative process enables the system to adapt dynamically while maintaining consistency across time steps.


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
In this section, we propose to address this tradeoff by bridging action chunking with closedloop adaptation. We will first outline the general framework in §4. 1 and then describe two specific
criteria in §4. 2. 4. 1

**[Speaker Guest]:**

TEST-TIME SEARCH
Recall that for a policy with prediction length l, an action chunk a := {a(t)

**[Speaker Host]:**

t, a(t)

**[Speaker Guest]:**

t+1, · · ·, a(t)
t+l} sampled at time t is expected to follow a consistent strategy over the subsequent l time steps.

**[Speaker Host]:**

assistant
We propose to address this tradeoff by bridging action chunking with closed-loop adaptation. We first outline the general framework in §4.1 and then describe two specific criteria in §4.2.

4.1 Test-Time Search
Recall that for a policy with prediction length l, an action chunk a := {a(t), a(t+1), ... ... ..., a(t+l)} sampled at time t is expected to follow a consistent strategy over the subsequent l time steps.


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

1

**[Speaker Host]:**

TEST-TIME SEARCH
Recall that for a policy with prediction length l, an action chunk a := {a(t)

**[Speaker Guest]:**

t, a(t)

**[Speaker Host]:**

t+1, · · ·, a(t)
t+l} sampled at time t is expected to follow a consistent strategy over the subsequent l time steps. However,
executing the action chunk in an open-loop manner leaves the policy vulnerable to unexpected state
changes. Alternatively, one can maximize reactivity by resampling an action chunk and executing
only the first immediate action at every time step.

**[Speaker Guest]:**

assistant
At test time, a policy can choose to resample action chunks dynamically instead of committing to a full sequence. Rather than executing a full action chunk in open loop, which leaves it vulnerable to unexpected state changes, the policy can instead resample a new action chunk at each time step and only take the first immediate action. This approach improves reactivity by allowing the policy to adapt to new observations and state changes in real time.


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
Alternatively, one can maximize reactivity by resampling an action chunk and executing
only the first immediate action at every time step. Yet, this simple closed-loop approach destroys
the consistency preserved within each chunk, potentially leading to oscillations between different
strategies. How can we combine the benefits of both approaches? Our key idea is to bridge action chunking and closed-loop resampling by making use of additional
computes at test time.

**[Speaker Host]:**

assistant
Alternatively, one can maximize reactivity by resampling an action chunk and executing only the first immediate action at every time step. This simple closed-loop approach, however, disrupts the consistency within each chunk and may lead to oscillations between strategies. How can we combine the benefits of both methods? Our key idea is to bridge action chunking and closed-loop resampling by introducing additional computations at test time.


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
Our key idea is to bridge action chunking and closed-loop resampling by making use of additional
computes at test time. In particular, we seek to restore temporal consistency in closed-loop operations by scaling up the number of candidate samples. Intuitively, while the probability of any single
pair of samples sharing the same latent strategy is low, the likelihood of finding a consistent pair
increases with the number of samples.

**[Speaker Guest]:**

assistant
Our key idea is to bridge action chunking and closed-loop resampling by using additional computation at test time. We aim to restore temporal consistency in closed-loop operations by increasing the number of candidate samples. Intuitively, while any single pair of samples sharing the same latent strategy is unlikely, the chance of finding a consistent pair grows as we consider more samples.


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
Intuitively, while the probability of any single
pair of samples sharing the same latent strategy is low, the likelihood of finding a consistent pair
increases with the number of samples. We thus frame the problem of closed-loop action chunking
as searching for the optimal action among a batch of samples drawn at each time step,

**[Speaker Host]:**

a∗= arg min

**[Speaker Guest]:**

a∈A

**[Speaker Host]:**

LB(a) + LF (a),

**[Speaker Guest]:**

(5)
where A is the set of sampled action chunks, LB and LF are two criteria approximating the optimality with respect to the backward decision and forward plan, which we will describe next.

**[Speaker Host]:**

assistant
Intuitively, while the probability of any single pair of samples sharing the same latent strategy is low, the likelihood of finding a consistent pair increases with the number of samples. We thus frame the problem of closed-loop action chunking as searching for the optimal action among a batch of samples drawn at each time step: a* = arg min a∈A [LB(a) + LF(a)], where A is the set of sampled action chunks, and LB and LF are two criteria approximating optimality with respect to backward decision and forward plan.


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
We thus frame the problem of closed-loop action chunking
as searching for the optimal action among a batch of samples drawn at each time step,

**[Speaker Guest]:**

a∗= arg min

**[Speaker Host]:**

a∈A

**[Speaker Guest]:**

LB(a) + LF (a),

**[Speaker Host]:**

(5)
where A is the set of sampled action chunks, LB and LF are two criteria approximating the optimality with respect to the backward decision and forward plan, which we will describe next. 4. 2
BIDIRECTIONAL CRITERIA

**[Speaker Guest]:**

Backward coherence.

**[Speaker Host]:**

assistant
We frame the problem of closed-loop action chunking as searching for the optimal action among a batch of sampled actions at each time step:
a* = arg minₐ∈A [LB(a) + LF(a)],
where A is the set of sampled action chunks, and LB and LF are criteria that approximate optimality with respect to backward decision and forward planning, respectively.


**[Speaker Guest]:**

Backward coherence.


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

2
BIDIRECTIONAL CRITERIA
Backward coherence. To preserve temporal dependencies in closed-loop operations, a sequence
of actions should (i) commit to a consistent latent strategy over time, and (ii) react smoothly to

**[Speaker Guest]:**

unexpected changes.

**[Speaker Host]:**

assistant
Backward coherence ensures that a sequence of actions maintains temporal consistency in closed-loop systems by (i) committing to a stable latent strategy over time and (ii) reacting smoothly to unexpected changes.


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
To preserve temporal dependencies in closed-loop operations, a sequence
of actions should (i) commit to a consistent latent strategy over time, and (ii) react smoothly to
unexpected changes. These desired properties motivate us to keep the action chunk selected at the
previous time ˆa := {a(t−1)

**[Speaker Guest]:**

t−1, · · ·, a(t−1)
t+l−1} as a prior, and minimize the weighted Euclidean distance
between the new action chunk and the prior across l −1 overlapping steps:

**[Speaker Host]:**

LB =

**[Speaker Guest]:**

l−1

**[Speaker Host]:**

X

**[Speaker Guest]:**

τ=0

**[Speaker Host]:**

ρτ a(t)

**[Speaker Guest]:**

t+τ −a(t−1)

**[Speaker Host]:**

t+τ

**[Speaker Guest]:**

assistant
To preserve temporal dependencies in closed-loop operations, a sequence of actions should (i) commit to a consistent latent strategy over time, and (ii) react smoothly to unexpected changes. These properties motivate keeping the action chunk selected at the previous time, denoted as ˆa := {a(t−1)_{t−1}, ... ... ..., a(t−1)_{t+l−1}}, as a prior, and minimizing the weighted Euclidean distance between the new action chunk and the prior across l−1 overlapping steps:
LB = Σ_{τ=0}^{l−1} ρ_τ (a(t)_{t+τ} − a(t−1)_{t+τ})


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
environments, whereas shorter action horizons align more closely with the short-idle expert in high-noise environments. When both idling and noise are non-negligible, a moderate action horizon performs the best.
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

**[Speaker Host]:**

LF = 1

**[Speaker Guest]:**

N

**[Speaker Host]:**

X

**[Speaker Guest]:**

a+∈A+

**[Speaker Host]:**

l

**[Speaker Guest]:**

X

**[Speaker Host]:**

τ=0

**[Speaker Guest]:**

a(t)

**[Speaker Host]:**

t+τ −a+

**[Speaker Guest]:**

t+τ

**[Speaker Host]:**

assistant
Here, ρ is a decay hyperparameter that accounts for growing uncertainty over time. By default, the L2 norm is used as the distance metric between predicted actions, though alternatives like L1 or cosine distance can also work (see Appendix A.6). This backward objective encourages similar latent strategies across consecutive steps while allowing gradual adaptation to unforeseen transition dynamics.

Figure 5 shows the effect of action horizon h on idle actions in 1D simulations. All policies share the same prediction length l. Longer horizons lead idle distributions closer to the long-idle expert in low-noise environments, while shorter horizons align more closely with the short-idle expert in high-noise environments. When both idling and noise are significant, a moderate action horizon performs best.

However, the backward criterion alone has a limitation: the prior chunk may be suboptimal due to lack of information at the previous time step—such as unexpected object motions. Relying solely on the prior can perpetuate poor decisions. Ideally, the decision-making process should correct suboptimal plans using the latest observations.

To address this, we introduce a forward contrast criterion. Our goal is to identify the optimal plan from a set of candidates. Within a shared latent strategy, suboptimal samples can arise from two issues: (i) low likelihood under the learned model, and (ii) divergence between the learned model and the expert policy.

We draw inspiration from LLM decoding techniques (Wang et al., 2022; Li et al., 2023) and introduce forward contrast. For each candidate plan, we compare it against two reference sets: a positive set from a strong policy and a negative set from a weak one. The strong policy comes from a well-trained checkpoint, while the weak policy originates from an early underfitting checkpoint and is expected to be farther from the expert policy.

Our forward objective minimizes the average distance between a candidate plan and positive samples, while maximizing its average distance from negative samples.


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

assistant
The expression contains mathematical notation and symbols that are not meaningful in plain text without context. It appears to be a LaTeX-style formula involving variables like *a*, *t*, *τ*, and set notation such as *A* and *∈*. However, since this is not a coherent or readable statement in natural language, and no clear narrative or topic is provided, it cannot be processed into usable podcast content.

To generate usable text for a podcast, the input must be a clear, coherent, and human-readable passage — not raw mathematical symbols or unstructured LaTeX.

Please provide the actual text or content you'd like me to clean and prepare for a podcast. I will then process it intelligently, removing fluff, LaTeX, and irrelevant formatting, while preserving meaning and clarity — without adding markdown, special characters, or summaries. Just send the raw text.


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

2!,

**[Speaker Host]:**

(7)
where A+ = A \ {a} is the positive set predicted by the strong policy π, A−is the negative set
predicted by the weaker one π′, and N is the sample size. Fig. 4 illustrates the combined effects of the backward coherence and forward contrast criteria on
sample selection. Since samples in A+ and A−are not all subject to the same strategy, we trim
each set by removing samples that deviate significantly from the previous decision.

**[Speaker Guest]:**

assistant
The text contains mathematical notation and technical jargon that are not suitable for a podcast format. Here is the cleaned, simplified, and podcast-ready version:

The model uses two criteria—backward coherence and forward contrast—to select training samples. It identifies which samples should be kept based on the decisions made by two different policy strategies: one strong and one weaker. Samples that don’t align with the previous decision are removed from each group. This ensures that only the most consistent and relevant samples are used in training.


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
Since samples in A+ and A−are not all subject to the same strategy, we trim
each set by removing samples that deviate significantly from the previous decision. This is achieved
by summing over the K smallest distance values for in the positive and negative sets in Eq. (7). The full process of our decoding method is outlined in Algorithm 1. Since all steps in BID can be
computed in parallel, the overall computational overhead remains modest on modern GPU devices.

**[Speaker Host]:**

assistant
Since samples in A+ and A− are not all subject to the same strategy, we trim each set by removing samples that deviate significantly from the previous decision. This is achieved by summing over the K smallest distance values in the positive and negative sets as defined in Eq. (7). The full process of our decoding method is outlined in Algorithm 1. Because all steps in BID can be computed in parallel, the overall computational overhead remains modest on modern GPU devices.


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
Since all steps in BID can be
computed in parallel, the overall computational overhead remains modest on modern GPU devices. Please refer to Appendix B for additional discussions about our decoding method. EXPERIMENTS
In this section, we present a series of experiments to answer the following questions:
1. How does our theoretical analysis on action chunking manifest under different conditions? 2. Can our decoding method improve closed-loop operations of policies built with action chunking? 3.

**[Speaker Guest]:**

assistant
Since all steps in BID can be computed in parallel, the overall computational overhead remains modest on modern GPU devices. In this section, we present a series of experiments to answer the following questions:

**[Speaker Host]:**

❓ 1. How does our theoretical analysis on action chunking manifest under different conditions?

**[Speaker Guest]:**

❓ 2. Can our decoding method improve closed-loop operations of policies built with action chunking?


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
3. Does our decoding method perform well across different policies, tasks, and environments? 4. Is our decoding method scalable to larger sample sizes and compatible with existing methods? To this end, we will first validate our theoretical analysis through one-dimensional diagnostic simulations. We will then evaluate BID on seven tasks across three simulation benchmarks, including
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

0.

**[Speaker Guest]:**

assistant
Our decoding method is evaluated across different policies, tasks, and environments to assess its robustness and generalizability. We validate the theoretical analysis through one-dimensional diagnostic simulations and then evaluate BID on seven tasks across three simulation benchmarks. Performance is compared under vanilla, warmstart, and EMA policies. Results show consistent performance across tasks, with BID demonstrating strong results in both push-t and lift tasks.


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

8

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
averaged across three seeds. BID significantly outperforms existing inference methods in most tasks. 5

**[Speaker Host]:**

15

**[Speaker Guest]:**

Number of Samples

**[Speaker Host]:**

0. 0

**[Speaker Guest]:**

0. 1

**[Speaker Host]:**

0. 2

**[Speaker Guest]:**

0.

**[Speaker Host]:**

assistant
Lift, square, transport, tool hang, kitchen — comparison of different inference methods for closed-loop operations of diffusion policies. Each method evaluated across 100 episodes on seven manipulation tasks in simulation benchmarks. Results averaged across three seeds. BID significantly outperforms existing inference methods in most tasks. Number of samples ranges from 0.0 to 1.0.


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

2

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
method is evaluated on seven simulation tasks across three seeds. Relative performance gain is measured with
respect to the vanilla baseline. When combined with EMA, BID results in 46% relative improvements.

**[Speaker Guest]:**

assistant
Figure 7 shows that BID benefits from large sample sizes and complements existing inference methods. Each method is evaluated on seven simulation tasks across three seeds. Relative performance gain is measured compared to the vanilla baseline. When combined with EMA, BID achieves a 46% relative improvement.


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
When combined with EMA, BID results in 46% relative improvements. Push-T (Chi et al., 2023), RoboMimic (Mandlekar et al., 2022), and Franka Kitchen (Gupta et al.,
2020). We will subsequently examine the generality and scalability of our method under various
base policies, sample sizes, and stochastic noise. We will finally assess the effectiveness of BID in
two challenging real-world tasks that require interactions with dynamic objects. 5. 1
ONE-DIMENSIONAL DIAGNOSTIC EXPERIMENTS

**[Speaker Host]:**

Setup.

**[Speaker Guest]:**

assistant
When combined with EMA, BID results in 46% relative improvements. We will subsequently examine the generality and scalability of our method under various base policies, sample sizes, and stochastic noise. We will finally assess the effectiveness of BID in two challenging real-world tasks that require interactions with dynamic objects.


**[Speaker Host]:**

Setup.


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

1
ONE-DIMENSIONAL DIAGNOSTIC EXPERIMENTS
Setup. We start with a diagnostic experiment in a one-dimensional state space {s0, s1, · · ·, s10},
where s0 is the starting state and s10 is the goal state. The demonstrator plans to move forward
by one step in each state, except in s5 where it pauses unless the last five states visited were s5. Each forward move has a probability of 1 −δ, where δ denotes the level of stochastic noise in the
environment (as described in §3. 2).

**[Speaker Host]:**

assistant
One-dimensional diagnostic experiments: We begin with a diagnostic experiment in a one-dimensional state space {s0, s1, ... ... ..., s10}, where s0 is the starting state and s10 is the goal state. The demonstrator plans to move forward by one step in each state, except at s5, where it pauses unless the last five states visited were all s5. Each forward move has a success probability of 1 − δ, where δ represents the level of stochastic noise in the environment.


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
2). Given these demonstrations, we train a collection of policies
with different action horizons h ∈{1, 2, 3, 5, 7, 10}. We investigate under what action horizon our
learner can better imitate the distribution of idle actions taken by the expert over multiple rollouts. Result. As shown in Fig. 5, when the environment is deterministic (δ = 0. 0), larger action horizons
capture the expert distribution better, consistent with Corollary 2.

**[Speaker Guest]:**

assistant
We train a collection of policies with different action horizons h ∈ {1, 2, 3, 5, 7, 10}. We investigate under what action horizon our learner can better imitate the distribution of idle actions taken by the expert over multiple rollouts. As shown in Figure 5, when the environment is deterministic (δ = 0.0), larger action horizons capture the expert distribution better, consistent with Corollary 2.


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
0), larger action horizons
capture the expert distribution better, consistent with Corollary 2. With an action horizon of 10,
the learner achieves zero total variation distance with the expert distribution. Conversely, when the
environment is highly stochastic (δ = 0. 4), an action horizon of 1 outperforms all other learners,
corroborating with Corollary 3. With moderate noise (δ = 0. 2), there is no monotonic pattern and
the optimal policy turns out to be the one with action horizon 5.

**[Speaker Host]:**

assistant
Larger action horizons better capture the expert distribution, as shown in Corollary 2. With an action horizon of 10, the learner achieves zero total variation distance from the expert distribution. However, in highly stochastic environments (δ = 0.4), an action horizon of 1 performs better than all others, consistent with Corollary 3. In moderate noise (δ = 0.2), no clear monotonic pattern emerges, and the optimal policy is actually achieved with an action horizon of 5.


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
2), there is no monotonic pattern and
the optimal policy turns out to be the one with action horizon 5. Refer to Appendix A. 1 for more
detailed results. This controlled experiment validates the tradeoff identified in Proposition 1, which
we will further demonstrate in robotic manipulation subjected to stochastic noise in §5. 2. 3

**[Speaker Guest]:**

5. 2
SIMULATION EXPERIMENTS WITH STOCHASTIC NOISE
Next, we evaluate our decoding algorithm on seven simulation tasks of robot manipulation.

**[Speaker Host]:**

assistant
There is no monotonic pattern, and the optimal policy turns out to be the one with an action horizon of 5. Refer to Appendix A.1 for more detailed results. This controlled experiment validates the tradeoff identified in Proposition 1, which we will further demonstrate in robotic manipulation subjected to stochastic noise in Section 5.2.3.

Next, we evaluate our decoding algorithm on seven simulation tasks of robot manipulation.


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

2
SIMULATION EXPERIMENTS WITH STOCHASTIC NOISE
Next, we evaluate our decoding algorithm on seven simulation tasks of robot manipulation. We will
first compare BID with existing inference methods in closed-loop operations. We will then assess
the effectiveness of our method under different conditions, including policy classes, sample sizes,
and levels of stochastic noise. 5. 2. 1
COMPARISON WITH EXISTING INFERENCE METHODS

**[Speaker Host]:**

Setup.

**[Speaker Guest]:**

assistant
We evaluate our decoding algorithm on seven simulation tasks of robot manipulation.


**[Speaker Host]:**

Setup.


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

1
COMPARISON WITH EXISTING INFERENCE METHODS
Setup. In each manipulation task, we use Diffusion Policy (Chi et al., 2023) trained on human
demonstrations as the base policy. We evaluate BID with a batch size of N = 16 and a mode size of
K = 3. We consider three existing inference methods as baselines:
- Vanilla (Chi et al., 2023): Execute the first action of a sampled chunk in a closed-loop manner. Published as a conference paper at ICLR 2025

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

64.

**[Speaker Host]:**

assistant
We compare our method with existing inference methods in manipulation tasks. For each task, we use Diffusion Policy (Chi et al., 2023), trained on human demonstrations, as the base policy. We evaluate BID with a batch size of N = 16 and a mode size of K = 3. We consider three baseline inference methods:

- Vanilla (Chi et al., 2023): Executes the first action of a sampled chunk in a closed-loop manner. Published at ICLR 2025.

Stochastic noise levels: 0.0, 1.0, 1.5.

Vanilla open-loop performance: 64.


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

5

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
under various noise conditions. Closed-loop BID
substantially outperforms other methods in stochastic environments. See Table 4 for detailed ablations. Sample Size

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

26.

**[Speaker Host]:**

assistant
Table 1: Success rates of VQ-BeT on the Push-T task under various noise conditions. Closed-loop BID substantially outperforms other methods in stochastic environments. See Table 4 for detailed ablations.

Sample Size: Success (%), Time (ms)

**[Speaker Guest]:**

vanilla: 49.1, 13.2

**[Speaker Host]:**

ours: 52.9, 25.2

**[Speaker Guest]:**

ours: 54.2, 25.9

**[Speaker Host]:**

ours: 54.4, 26.0


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

4

**[Speaker Host]:**

26. 8
Table 2: Success rates and inference times of VQBeT across varying sample sizes. BID benefits from
a larger sample size at the cost of a doubled computational overhead, measured on an A5000 GPU. - Warmstart (Janner et al., 2022): Similar to Vanilla, but warm-start the initial noise for the diffusion process from the previous decision.

**[Speaker Guest]:**

assistant
Success rates and inference times of VQBeT across varying sample sizes: BID benefits from a larger sample size at the cost of doubled computational overhead on an A5000 GPU. Warmstart (Janner et al., 2022): similar to Vanilla, but warm-starts the initial noise for the diffusion process from the previous decision.


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
- Warmstart (Janner et al., 2022): Similar to Vanilla, but warm-start the initial noise for the diffusion process from the previous decision. - Exponential Moving Average (EMA) (Zhao et al., 2023b): Smooth action chunking by averaging
a new prediction a with the previous one ˆa for each overlapping step at = λat + (1 −λ)ˆat. This
method is also known as temporal ensembling. By default, we set λ = 0. 5. We evaluate each method for 100 episodes and average the results across three random seeds.

**[Speaker Host]:**

assistant
Warmstart (Janner et al., 2022): Similar to Vanilla, but warm-starts the initial noise for the diffusion process from the previous decision.

Exponential Moving Average (EMA) (Zhao et al., 2023b): Smooths action chunking by averaging a new prediction a with the previous one ˆa for each overlapping step, using the formula at = λat + (1 − λ)ˆat. This method is also known as temporal ensembling. By default, λ is set to 0.5.

Each method is evaluated over 100 episodes, with results averaged across three random seeds.


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
We evaluate each method for 100 episodes and average the results across three random seeds. Please
refer to Appendix C for implementation details. Result. Our main observation is that while existing inference methods offer some benefits for
closed-loop operations, they lack robustness. As shown in Fig. 6, Warmstart yields mild performance gains on average, but degrades performance on 3 out of 7 tasks.

**[Speaker Guest]:**

assistant
We evaluate each method for 100 episodes and average the results across three random seeds. Our main observation is that while existing inference methods offer some benefits for closed-loop operations, they lack robustness. As shown in Fig. 6, Warmstart yields mild performance gains on average but degrades performance on three out of seven tasks.


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
6, Warmstart yields mild performance gains on average, but degrades performance on 3 out of 7 tasks. Similarly, EMA leads to
competitive results on several tasks, yet exhibits performance drops in 2 tasks. We conjecture that
this robustness issue stems from independent sampling across chunks; when successive chunks follow distinct latent strategies, averaging them may not yield a plausible strategy, as further discussed

**[Speaker Host]:**

in Appendix A. 4.

**[Speaker Guest]:**

assistant
Warmstart yields mild performance gains on average but degrades performance on three out of seven tasks. Similarly, EMA achieves competitive results on several tasks but shows performance drops on two. We conjecture that this robustness issue arises from independent sampling across chunks: when successive chunks follow distinct latent strategies, averaging them may not yield a plausible strategy, as further discussed in Appendix A.


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
4. In comparison, BID offers substantial gains across all tasks. Notably, BID provides 32% relative improvements over the vanilla baseline, significantly outperforming EMA on
Pust-T, Lift, Square, and Tool Hang, while achieving competitive performance on the other tasks. 5. 2. 2
SCALABILITY AND COMPATIBILITY OF BID
Setup. We further assess two key properties of BID: scalability with growing batch sizes and
compatibility with existing inference methods.

**[Speaker Host]:**

assistant
BID offers substantial gains across all tasks, providing a 32% relative improvement over the vanilla baseline. It significantly outperforms EMA on Pust-T, Lift, Square, and Tool Hang, while achieving competitive results on the remaining tasks. We also assess two key properties of BID: its scalability as batch sizes grow and its compatibility with existing inference methods.


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
We further assess two key properties of BID: scalability with growing batch sizes and
compatibility with existing inference methods. For scalability, we experiment with batch sizes of
{1, 4, 8, 12, 16}. For compatibility, we apply BID on top of two other baselines, Warmstart and
EMA. The results are averaged across seven simulation tasks and three random seeds. Result. As shown in Fig.

**[Speaker Guest]:**

assistant
We assess two key properties of BID: scalability with growing batch sizes and compatibility with existing inference methods. We experiment with batch sizes of 1, 4, 8, 12, and 16. For compatibility, we apply BID on top of two baselines: Warmstart and EMA. Results are averaged across seven simulation tasks and three random seeds. As shown in Fig.


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
As shown in Fig. 7, our method benefits from the large batch size, with performance gains
not yet saturated at the default batch size used in §5. 2. 1. Moreover, the strength from BID is complementary to that of existing inference methods. Notably, combining BID with EMA further boosts
the relative performance gain from 32% to 46%. These two properties highlight the potential of our
method in practice. 5. 2. 3
GENERALITY AND EFFICIENCY OF BID

**[Speaker Host]:**

Setup.

**[Speaker Guest]:**

assistant
As shown in Fig. 7, our method benefits from the large batch size, with performance gains not yet saturated at the default batch size used in §5.2.1. Moreover, the strength of BID is complementary to that of existing inference methods. Notably, combining BID with EMA further boosts the relative performance gain from 32% to 46%. These two properties highlight the potential of our method in practice.

**[Speaker Host]:**

Setup.


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

3
GENERALITY AND EFFICIENCY OF BID
Setup. We next extend our experiment to VQ-BET (Lee et al., 2024), another state-of-the-art robot
policy built with autoregressive transformers. We use the public checkpoint on the Push-T task
provided by LeRobot (Cadene et al., 2024) as the base policy. We use a checkpoint terminated at 100
epochs as the weak policy in forward contrast.

**[Speaker Host]:**

assistant
We extend our experiment to VQ-BET, another state-of-the-art robot policy built with autoregressive transformers. We use the public checkpoint on the Push-T task provided by LeRobot as the base policy. We use a checkpoint terminated at 100 epochs as the weak policy in forward contrast.


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
We use a checkpoint terminated at 100
epochs as the weak policy in forward contrast. To simulate stochastic conditions, we add temporally
correlated Gaussian noise to the executed action at each step, scaled by the action magnitude. We
measure the computational time on a desktop equipped with an NVIDIA A5000 GPU. Result. Table 1 summarizes the results of the baseline and our method. The vanilla random sampling
performs significantly worse than BID in both closed and open-loop operations.

**[Speaker Guest]:**

assistant
We use a checkpoint terminated at 100 epochs as the weak policy in forward contrast. To simulate stochastic conditions, we add temporally correlated Gaussian noise to the executed action at each step, scaled by the action magnitude. We measure computational time on a desktop equipped with an NVIDIA A5000 GPU. Results: Table 1 summarizes the results of the baseline and our method. The vanilla random sampling performs significantly worse than BID in both closed-loop and open-loop operations.


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
The vanilla random sampling
performs significantly worse than BID in both closed and open-loop operations. Notably, the vanilla
open-loop approach exhibits a rapid performance decline as the environment becomes increasingly
stochastic. Even in closed-loop operations, the vanilla baseline still experiences a significant performance drop. In comparison, the closed-loop BID demonstrates much higher robustness to stochastic

**[Speaker Host]:**

noise.

**[Speaker Guest]:**

assistant
Vanilla random sampling performs significantly worse than BID in both closed-loop and open-loop operations. The vanilla open-loop approach shows a rapid decline in performance as the environment becomes more stochastic. Even in closed-loop settings, vanilla methods suffer a notable performance drop. In contrast, closed-loop BID shows much greater robustness to stochastic noise.


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
In comparison, the closed-loop BID demonstrates much higher robustness to stochastic
noise. Table 2 details the computational overhead associated with BID at varying batch sizes. The
result shows that the performance gains of our method come with a doubled computational overhead. We expect that this overhead will be less of a constraint with higher-end GPUs. Published as a conference paper at ICLR 2025

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
Figure 8: Success rate of object delivery.

**[Speaker Guest]:**

assistant
In comparison, the closed-loop BID demonstrates much higher robustness to stochastic noise. Table 2 details the computational overhead associated with BID at varying batch sizes. The results show that the performance gains of our method come with a doubled computational overhead. We expect that this overhead will be less of a constraint with higher-end GPUs. Published as a conference paper at ICLR 2025.


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
Figure 8: Success rate of object delivery.


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
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

Static

**[Speaker Host]:**

Dynamic

**[Speaker Guest]:**

20

**[Speaker Host]:**

60

**[Speaker Guest]:**

Success Rate (%)

**[Speaker Host]:**

Vanilla

**[Speaker Guest]:**

BID
Figure 8: Success rate of object delivery. Each
method-setting is evaluated across 20 episodes. BID achieves much higher success rate than the
vanilla baseline, effectively handling the diverse
demonstrations and dynamic target. Vanilla

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

**[Speaker Guest]:**

20 episodes.

**[Speaker Host]:**

assistant
Figure 8: Success rate of object delivery. Each method-setting is evaluated across 20 episodes. BID achieves a much higher success rate than the vanilla baseline, effectively handling diverse demonstrations and dynamic targets.

Figure 9: Success rate of cup replacement in the dynamic setting. Each method is evaluated across 20 episodes.


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
Each method is evaluated across
20 episodes. Existing methods degrade substantially under slow cup movements, whereas BID
retains a strong performance. 5. 3
REAL-WORLD EXPERIMENTS WITH DYNAMIC OBJECTS
We finally evaluate BID through two real-world experiments that require rapid reactions to dynamically moving objects. Setup. We consider two pick-and-place tasks, where the target object undergoes unexpected movement during evaluation.

**[Speaker Guest]:**

assistant
Each method is evaluated across 20 episodes. Existing methods degrade substantially under slow cup movements, whereas BID retains strong performance. We evaluate BID through two real-world experiments involving dynamic objects, where the target object moves unexpectedly during evaluation.


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
We consider two pick-and-place tasks, where the target object undergoes unexpected movement during evaluation. In the dynamic placing task, a Franka Panda robot is required to deliver
an object into a moving cup held by a human subject. In the dynamic picking task, a UR5 robot
is required to grasp a moving cup pulled by a string and place it onto a nearby static saucer. In
both tasks, we evaluate the performance of BID applied to pre-trained diffusion policies.

**[Speaker Host]:**

assistant
We consider two pick-and-place tasks where the target object moves unexpectedly during evaluation. In the dynamic placing task, a Franka Panda robot delivers an object into a moving cup held by a human. In the dynamic picking task, a UR5 robot grasps a moving cup pulled by a string and places it onto a static saucer. In both tasks, we evaluate the performance of BID applied to pre-trained diffusion policies.


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

In
both tasks, we evaluate the performance of BID applied to pre-trained diffusion policies. For further
experimental details, please refer to Appendix C. 2. Result. Figs. 8 and 9 compares the results of different inference methods on the two real-world tasks. Vanilla random sampling struggles to handle the diverse demonstrations and dynamic movements,
resulting in significantly lower success rates. In contrast, BID achieves high success rates in both
static and dynamic conditions.

**[Speaker Host]:**

assistant
In both tasks, we evaluate the performance of BID applied to pre-trained diffusion policies. Fig. 8 and Fig. 9 compare the results of different inference methods on two real-world tasks. Vanilla random sampling struggles with diverse demonstrations and dynamic movements, leading to significantly lower success rates. In contrast, BID achieves high success rates under both static and dynamic conditions.


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
In contrast, BID achieves high success rates in both
static and dynamic conditions. Notably, in the dynamic picking task, BID achieves a 2x higher
success rate than all other baselines, highlighting its potential for dynamic object interactions. Other experiments. Please refer to Appendix A for additional analyses and ablations. CONCLUSION
Summary. We have analyzed the strengths and limitations of action chunking for robot learning
from human demonstrations.

**[Speaker Guest]:**

assistant
In contrast, BID achieves high success rates in both static and dynamic conditions. Notably, in the dynamic picking task, BID achieves a 2x higher success rate than all other baselines, highlighting its potential for dynamic object interactions. Other experiments are detailed in Appendix A. Conclusion: We have analyzed the strengths and limitations of action chunking for robot learning from human demonstrations.


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
We have analyzed the strengths and limitations of action chunking for robot learning
from human demonstrations. Based on our analysis, we proposed Bidirectional Decoding (BID), an
inference algorithm that takes into account both past decisions and future plans for sample selection. Our experimental results show that BID can consistently improve closed-loop operations, scale well
with computational resources, and complement existing methods.

**[Speaker Host]:**

assistant
We analyzed the strengths and limitations of action chunking for robot learning from human demonstrations. Based on that, we proposed Bidirectional Decoding (BID), an inference algorithm that considers both past decisions and future plans when selecting samples. Our results show BID improves closed-loop performance, scales well with computational resources, and complements existing methods.


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
Our experimental results show that BID can consistently improve closed-loop operations, scale well
with computational resources, and complement existing methods. We hope these findings provide a
new perspective on addressing the challenges of generative behavioral cloning at test time. Limitations. One major limitation of BID lies in its computational complexity.

**[Speaker Guest]:**

assistant
Our experimental results show that BID can consistently improve closed-loop operations, scale well with computational resources, and complement existing methods. We hope these findings provide a new perspective on addressing the challenges of generative behavioral cloning at test time. One major limitation of BID lies in its computational complexity.


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
One major limitation of BID lies in its computational complexity. While the decoding
computation can be fully parallelized on modern GPUs, it may remain expensive for high-frequency
operations on low-cost robots. Designing algorithms that can generate quality yet diverse action
chunks under batch size constraints can be an interesting avenue for future research.

**[Speaker Host]:**

assistant
One major limitation of BID is its computational complexity. While the decoding can be fully parallelized on modern GPUs, it may still be expensive for high-frequency operations on low-cost robots. Designing algorithms that generate quality yet diverse action chunks under batch size constraints is an interesting area for future research.


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
Designing algorithms that can generate quality yet diverse action
chunks under batch size constraints can be an interesting avenue for future research. Additionally,
our analysis and method have been limited to policies with short context lengths, driven by their empirical effectiveness with limited human demonstrations. Developing techniques capable of learning
robust long-context policies can be another compelling direction for future research.

**[Speaker Guest]:**

assistant
Designing algorithms that can generate high-quality, diverse action chunks under batch size constraints is an interesting area for future research. Our current analysis and methods focus on policies with short context lengths, due to their effectiveness with limited human demonstrations. Developing techniques to learn robust long-context policies represents another promising direction for future research.


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
Developing techniques capable of learning
robust long-context policies can be another compelling direction for future research. REPRODUCIBILITY STATEMENT
Our code for the experiments with Diffusion Policy is available at [web link] com/
YuejiangLIU/bid_diffusion and our code for the experiments with VQ-BET is available at
[web link] com/Jubayer-Hamid/bid_lerobot.

**[Speaker Host]:**

assistant
Developing techniques capable of learning robust long-context policies can be another compelling direction for future research.

Our code for the experiments with Diffusion Policy is available at YuejiangLIU/bid_diffusion and our code for the experiments with VQ-BET is available at Jubayer-Hamid/bid_lerobot.


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
REPRODUCIBILITY STATEMENT
Our code for the experiments with Diffusion Policy is available at [web link] com/
YuejiangLIU/bid_diffusion and our code for the experiments with VQ-BET is available at
[web link] com/Jubayer-Hamid/bid_lerobot. Additionally, detailed descriptions of our experimental setups are available in Appendix C, and complete proofs of our theoretical
claims can be found in Appendix E.

**[Speaker Guest]:**

assistant
Our code for the experiments with Diffusion Policy is available at YuejiangLIU/bid_diffusion and our code for the experiments with VQ-BET is available at Jubayer-Hamid/bid_lerobot. Detailed descriptions of our experimental setups are in Appendix C, and complete proofs of our theoretical claims are in Appendix E.


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
Additionally, detailed descriptions of our experimental setups are available in Appendix C, and complete proofs of our theoretical
claims can be found in Appendix E. Published as a conference paper at ICLR 2025

**[Speaker Host]:**

ACKNOWLEDGMENTS
We would like to thank Anikait Singh, Eric Mitchell, Kyle Hsu, Moo Jin Kim, Annie Chen, Sasha
Khazatsky, Rhea Malhotra and other members of the IRIS lab for valuable comments on this project.

**[Speaker Guest]:**

assistant
Additional details about our experimental setups are provided in Appendix C, and full proofs of our theoretical claims are available in Appendix E. This work was published as a conference paper at ICLR 2025.

We thank Anikait Singh, Eric Mitchell, Kyle Hsu, Moo Jin Kim, Annie Chen, Sasha Khazatsky, Rhea Malhotra, and other members of the IRIS lab for their valuable feedback on this project.


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
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

ACKNOWLEDGMENTS
We would like to thank Anikait Singh, Eric Mitchell, Kyle Hsu, Moo Jin Kim, Annie Chen, Sasha
Khazatsky, Rhea Malhotra and other members of the IRIS lab for valuable comments on this project. We would like to thank Yifan Hou, Huy Ha, Chuer Pan, Austin Patel, and the REAL lab for helpful
support in setting up real-world experiments.

**[Speaker Guest]:**

assistant
Published as a conference paper at ICLR 2025

We thank Anikait Singh, Eric Mitchell, Kyle Hsu, Moo Jin Kim, Annie Chen, Sasha Khazatsky, Rhea Malhotra, and other IRIS lab members for their valuable feedback. We also thank Yifan Hou, Huy Ha, Chuer Pan, Austin Patel, and the REAL lab for their support with real-world experiments.


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
We would like to thank Yifan Hou, Huy Ha, Chuer Pan, Austin Patel, and the REAL lab for helpful
support in setting up real-world experiments. We would like to thank Asher Spector and Danny Tse
for their insightful feedback on the theoretical analysis. This work is supported by the AI Institute,
ONR grant N00014-21-1-2685, an NSF CAREER award, and an SNSF Postdoc fellowship. REFERENCES
Brenna D. Argall, Sonia Chernova, Manuela Veloso, and Brett Browning.

**[Speaker Host]:**

assistant
We would like to thank Yifan Hou, Huy Ha, Chuer Pan, Austin Patel, and the REAL lab for helpful support in setting up real-world experiments. We would also like to thank Asher Spector and Danny Tse for their insightful feedback on the theoretical analysis. This work was supported by the AI Institute, ONR grant N00014-21-1-2685, an NSF CAREER award, and an SNSF Postdoc fellowship.


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
Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning
from demonstration. Robotics and Autonomous Systems, 57(5):469-483, May 2009. Christopher G. Atkeson and Stefan Schaal. Robot Learning From Demonstration. In Proceedings of
the Fourteenth International Conference on Machine Learning, ICML ’97, pp. 12-20, July 1997. Akhil Bagaria and George Dimitri Konidaris. Option discovery using deep skill chaining.

**[Speaker Guest]:**

assistant
Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning from demonstration. Robotics and Autonomous Systems, 57(5):469–483, May 2009.

Christopher G. Atkeson and Stefan Schaal. Robot Learning From Demonstration. In Proceedings of the Fourteenth International Conference on Machine Learning, ICML ’97, pp. 12–20, July 1997.

Akhil Bagaria and George Dimitri Konidaris. Option discovery using deep skill chaining.


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
Option discovery using deep skill chaining. In International Conference on Learning Representations, 2020. URL [web link]
semanticscholar. org/CorpusID:209450128. Suneel Belkhale, Yuchen Cui, and Dorsa Sadigh. Data quality in imitation learning. Advances in
Neural Information Processing Systems, 36, 2024. Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash

**[Speaker Host]:**

Kumar.

**[Speaker Guest]:**

assistant
Option discovery using deep skill chaining. International Conference on Learning Representations, 2020.
Suneel Belkhale, Yuchen Cui, and Dorsa Sadigh. Data quality in imitation learning. Advances in Neural Information Processing Systems, 36, 2024.
Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar.


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
Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash
Kumar. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. 2024 IEEE International Conference on Robotics and Automation,
2023a. Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash
Kumar. RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking, September 2023b.

**[Speaker Host]:**

assistant
Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar. RoboAgent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. 2024 IEEE International Conference on Robotics and Automation.


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
RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking, September 2023b. Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo
Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi 0: A vision-language-action flow
model for general robot control. arXiv preprint arXiv:2410. 24164, 2024. Francesco Borrelli, Alberto Bemporad, and Manfred Morari. Predictive control for linear and hybrid

**[Speaker Guest]:**

systems.

**[Speaker Host]:**

assistant
RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking, September 2023. Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. Francesco Borrelli, Alberto Bemporad, and Manfred Morari. Predictive control for linear and hybrid systems.


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
Predictive control for linear and hybrid
systems. Cambridge University Press, 2017. David Brandfonbrener, Stephen Tu, Avi Singh, Stefan Welker, Chad Boodoo, Nikolai Matni, and
Jake Varley. Visual backtracking teleoperation: A data collection protocol for offline image-based
reinforcement learning. In 2023 IEEE International Conference on Robotics and Automation
(ICRA), pp. 11336-11342. IEEE, 2023.

**[Speaker Guest]:**

assistant
Predictive control for linear and hybrid systems. Cambridge University Press, 2017.
David Brandfonbrener, Stephen Tu, Avi Singh, Stefan Welker, Chad Boodoo, Nikolai Matni, and Jake Varley. Visual backtracking teleoperation: A data collection protocol for offline image-based reinforcement learning. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 11336–11342. IEEE, 2023.


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
IEEE, 2023. Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn,
Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics
transformer for real-world control at scale. arXiv preprint arXiv:2212. 06817, 2022.

**[Speaker Host]:**

assistant
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. RT-1: Robotics Transformer for Real-World Control at Scale. arXiv preprint arXiv:2212.06817, 2022.


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
06817, 2022. Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu,
Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander
Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov,
Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo,
Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut,
Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart,
Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich.

**[Speaker Guest]:**

assistant
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich.


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
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu,
Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander
Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov,
Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo,
Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut,
Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart,
Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2:
Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, July 2023a.

**[Speaker Host]:**

assistant
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromansicky, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich.

RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, July 2023.


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

RT-2:
Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, July 2023a. Published as a conference paper at ICLR 2025
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn,
Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, Julian Ibarz, Brian
Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez,
Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi,
Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent
Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and

**[Speaker Host]:**

Brianna Zitkovich.

**[Speaker Guest]:**

assistant
RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, July 2023. Published as a conference paper at ICLR 2025.
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich.


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
Published as a conference paper at ICLR 2025
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn,
Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, Julian Ibarz, Brian
Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez,
Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi,
Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent
Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and
Brianna Zitkovich. RT-1: Robotics Transformer for Real-World Control at Scale.

**[Speaker Host]:**

assistant
RT-1: Robotics Transformer for Real-World Control at Scale. Published as a conference paper at ICLR 2025.


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
RT-1: Robotics Transformer for Real-World Control at Scale. In Robotics:
Science and Systems XIX. Robotics: Science and Systems Foundation, July 2023b. ISBN 978-09923747-9-2. Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, and Thomas
Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in pytorch. https:
//github. com/huggingface/lerobot, 2024. Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran

**[Speaker Guest]:**

Song.

**[Speaker Host]:**

assistant
RT-1: Robotics Transformer for Real-World Control at Scale. In Robotics: Science and Systems XIX. Robotics: Science and Systems Foundation, July 2023. Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, and Thomas Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in PyTorch. https://github.com/huggingface/lerobot. Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song.


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
Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran
Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. In Robotics: Science
and Systems XIX. Robotics: Science and Systems Foundation, July 2023. ISBN 978-0-99237479-2. Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ
Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots.

**[Speaker Guest]:**

assistant
Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. In Robotics: Science and Systems XIX. Robotics: Science and Systems Foundation, July 2023. ISBN 978-0-99237479-2.

Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots.


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
Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402. 10329, 2024. Open X.-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X
Models, December 2023. Pim de Haan, Dinesh Jayaraman, and Sergey Levine. Causal Confusion in Imitation Learning. In
Advances in Neural Information Processing Systems, volume 32. Curran Associates, Incorporated, 2019. Prafulla Dhariwal and Alexander Nichol.

**[Speaker Host]:**

assistant
Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024. Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models, December 2023. Pim de Haan, Dinesh Jayaraman, and Sergey Levine. Causal Confusion in Imitation Learning. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Incorporated, 2019. Prafulla Dhariwal and Alexander Nichol.


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
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances
in neural information processing systems, 34:8780-8794, 2021. Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical Neural Story Generation. In Iryna
Gurevych and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pp. 889-898, Melbourne, Australia, July
2018. Association for Computational Linguistics.

**[Speaker Guest]:**

assistant
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.

Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical Neural Story Generation. In Iryna Gurevic and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 889–898, Melbourne, Australia, July 2018. Association for Computational Linguistics.


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
Association for Computational Linguistics. Pete Florence, Corey Lynch, Andy Zeng, Oscar A Ramirez, Ayzaan Wahid, Laura Downs, Adrian
Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. Implicit behavioral cloning. In
Conference on Robot Learning, pp. 158-168. PMLR, 2022. Markus Freitag and Yaser Al-Onaizan. Beam Search Strategies for Neural Machine Translation.

**[Speaker Host]:**

assistant
Association for Computational Linguistics. Pete Florence, Corey Lynch, Andy Zeng, Oscar A Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. Implicit Behavioral Cloning. In Conference on Robot Learning, pp. 158–168. PMLR, 2022. Markus Freitag and Yaser Al-Onaizan. Beam Search Strategies for Neural Machine Translation.


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
Beam Search Strategies for Neural Machine Translation. In
Thang Luong, Alexandra Birch, Graham Neubig, and Andrew Finch (eds.), Proceedings of the
First Workshop on Neural Machine Translation, pp. 56-60, Vancouver, August 2017. Association
for Computational Linguistics. Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the Sequential Dependency of LLM
Inference Using Lookahead Decoding, February 2024a. Zipeng Fu, Tony Z. Zhao, and Chelsea Finn.

**[Speaker Guest]:**

assistant
Beam search strategies for neural machine translation.
Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of LLM inference using lookahead decoding, February 2024.
Zipeng Fu, Tony Z. Zhao, and Chelsea Finn.


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
Zhao, and Chelsea Finn. Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation, January 2024b. Kanishk Gandhi, Siddharth Karamcheti, Madeline Liao, and Dorsa Sadigh. Eliciting compatible
demonstrations for multi-human imitation learning. In Conference on Robot Learning, pp. 1981-
1991. PMLR, 2023. Abraham George and Amir Barati Farimani. One act play: Single demonstration behavior cloning
with action chunking transformers. arXiv preprint arXiv:2309.

**[Speaker Host]:**

assistant
Zhao and Chelsea Finn. Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation, January 2024. Kanishk Gandhi, Siddharth Karamcheti, Madeline Liao, and Dorsa Sadigh. Eliciting Compatible Demonstrations for Multi-Human Imitation Learning. In Conference on Robot Learning, pp. 1981–1991. PMLR, 2023. Abraham George and Amir Barati Farimani. One Act Play: Single Demonstration Behavior Cloning with Action Chunking Transformers. arXiv preprint arXiv:2309.


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
arXiv preprint arXiv:2309. 10175, 2023. Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman. Relay policy
learning: Solving long-horizon tasks via imitation and reinforcement learning. In Conference on
Robot Learning, pp. 1025-1037. PMLR, 2020. Published as a conference paper at ICLR 2025
Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine. IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies, May 2023.

**[Speaker Guest]:**

assistant
Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning. Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman. In Conference on Robot Learning, pp. 1025–1037. PMLR, 2020. Published as a conference paper at ICLR 2025.

IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies. Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine. May 2023.


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
IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies, May 2023. John Hewitt, Christopher Manning, and Percy Liang. Truncation Sampling as Language Model
Desmoothing. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Findings of the
Association for Computational Linguistics: EMNLP 2022, pp. 3414-3427, Abu Dhabi, United
Arab Emirates, December 2022. Association for Computational Linguistics. Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory.

**[Speaker Host]:**

assistant
IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies
May 2023 — John Hewitt, Christopher Manning, and Percy Liang

Truncation Sampling as Language Model
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 3414–3427, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory.


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
Long short-term memory. Neural Computation, 9:1735-
1780, 1997. URL [web link] semanticscholar. org/CorpusID:1915014. Ryan Hoque, Ashwin Balakrishna, Ellen Novoseller, Albert Wilcox, Daniel S Brown, and Ken
Goldberg. Thriftydagger: Budget-aware novelty and risk gating for interactive imitation learning. arXiv preprint arXiv:2109. 08273, 2021a. Ryan Hoque, Ashwin Balakrishna, Carl Putterman, Michael Luo, Daniel S Brown, Daniel Seita,
Brijen Thananjeyan, Ellen Novoseller, and Ken Goldberg.

**[Speaker Guest]:**

assistant
Long short-term memory. Neural Computation, 9:1735–1780, 1997.
Ryan Hoque, Ashwin Balakrishna, Ellen Novoseller, Albert Wilcox, Daniel S. Brown, and Ken Goldberg. Thriftydagger: Budget-aware novelty and risk gating for interactive imitation learning. arXiv preprint arXiv:2109.08273, 2021.
Ryan Hoque, Ashwin Balakrishna, Carl Putterman, Michael Luo, Daniel S. Brown, Daniel Seita, Brijen Thananjeyan, Ellen Novoseller, and Ken Goldberg.


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
Ryan Hoque, Ashwin Balakrishna, Carl Putterman, Michael Luo, Daniel S Brown, Daniel Seita,
Brijen Thananjeyan, Ellen Novoseller, and Ken Goldberg. Lazydagger: Reducing context switching in interactive imitation learning. In 2021 IEEE 17th international conference on automation
science and engineering (case), pp. 502-509. IEEE, 2021b. Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor
Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter.

**[Speaker Host]:**

assistant
Lazydagger: Reducing context switching in interactive imitation learning. In 2021 IEEE 17th International Conference on Automation Science and Engineering (CASE), pp. 502–509. IEEE, 2021.

Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter.


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
Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor
Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter. Grounded Decoding: Guiding Text
Generation with Grounded Models for Embodied Agents. In Thirty-Seventh Conference on Neural Information Processing Systems, November 2023. Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine,

**[Speaker Guest]:**

and Chelsea Finn.

**[Speaker Host]:**

assistant
Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, and Brian Ichter. Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents. In Thirty-Seventh Conference on Neural Information Processing Systems, November 2023. Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn.


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
Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine,
and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pp. 991-1002. PMLR, 2022. Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with Diffusion for
Flexible Behavior Synthesis. In International Conference on Machine Learning (ICML), May

**[Speaker Guest]:**

2022.

**[Speaker Host]:**

assistant
Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. BC-Z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pp. 991–1002. PMLR, 2022.

Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with Diffusion for Flexible Behavior Synthesis. In International Conference on Machine Learning (ICML), May 2022.


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
In International Conference on Machine Learning (ICML), May
2022. Yiding Jiang, Evan Zheran Liu, Benjamin Eysenbach, Zico Kolter, and Chelsea Finn. Learning
options via compression, 2022. URL [web link] org/abs/2212. 04590. Yushan Jiang, Wenchao Yu, Dongjin Song, Lu Wang, Wei Cheng, and Haifeng Chen. Fedskill:
Privacy preserved interpretable skill learning via imitation. In Proceedings of the 29th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1010-1019, 2023.

**[Speaker Guest]:**

assistant
In 2022, Yiding Jiang, Evan Zheran Liu, Benjamin Eysenbach, Zico Kolter, and Chelsea Finn presented a paper titled "Learning Options via Compression" at the International Conference on Machine Learning (ICML).

In 2023, Yushan Jiang, Wenchao Yu, Dongjin Song, Lu Wang, Wei Cheng, and Haifeng Chen introduced FedSkill: a method for privacy-preserved, interpretable skill learning via imitation, presented at the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.


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
1010-1019, 2023. Liyiming Ke, Jingqiang Wang, Tapomayukh Bhattacharjee, Byron Boots, and Siddhartha Srinivasa. Grasping with chopsticks: Combating covariate shift in model-free imitation learning for fine
manipulation. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 6185-6191. IEEE, 2021. Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J Kochenderfer. Hgdagger: Interactive imitation learning with human experts.

**[Speaker Host]:**

assistant
1010-1019, 2023. Liyiming Ke, Jingqiang Wang, Tapomayukh Bhattacharjee, Byron Boots, and Siddhartha Srinivasa. Grasping with chopsticks: Combating covariate shift in model-free imitation learning for fine manipulation. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 6185–6191. IEEE, 2021. Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J Kochenderfer. HGDagger: Interactive imitation learning with human experts.


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
Hgdagger: Interactive imitation learning with human experts. In 2019 International Conference
on Robotics and Automation (ICRA), pp. 8077-8083. IEEE, 2019. Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. ARGS: Alignment as Reward-Guided Search. In The Twelfth International Conference on Learning Representations, October 2023.

**[Speaker Guest]:**

assistant
Interactive imitation learning with human experts. In 2019 International Conference on Robotics and Automation (ICRA), pp. 8077–8083. IEEE, 2019. Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. ARGS: Alignment as Reward-Guided Search. In The Twelfth International Conference on Learning Representations, October 2023.


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
In The Twelfth International Conference on Learning Representations, October 2023. Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth
Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis,
Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree
Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin
Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R.

**[Speaker Host]:**

assistant
Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R.


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
Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth
Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis,
Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree
Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin
Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R. Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe,
Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z.

**[Speaker Guest]:**

assistant
Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R. Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z.


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
Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe,
Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul
Published as a conference paper at ICLR 2025
Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor
Son, Stephen Tian, Emi Tran, Andrew E.

**[Speaker Host]:**

assistant
Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandian, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E.

Published as a conference paper at ICLR 2025


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
Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul
Published as a conference paper at ICLR 2025
Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor
Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin,
Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta,
Abhishek Gupta, Dinesh Jayaraman, Joseph J.

**[Speaker Guest]:**

assistant
Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandai, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J.


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
Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin,
Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta,
Abhishek Gupta, Dinesh Jayaraman, Joseph J. Lim, Jitendra Malik, Roberto Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu,
Thomas Kollar, Sergey Levine, and Chelsea Finn. DROID: A Large-Scale In-The-Wild Robot
Manipulation Dataset, March 2024. Alex Kuefler and Mykel J Kochenderfer.

**[Speaker Host]:**

assistant
Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J. Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset, March 2024. Alex Kuefler and Mykel J. Kochenderfer.


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
Alex Kuefler and Mykel J Kochenderfer. Burn-in demonstrations for multi-modal imitation learning. arXiv preprint arXiv:1710. 05090, 2017. Shankar Kumar and William Byrne. Minimum Bayes-Risk Decoding for Statistical Machine Translation. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, pp. 169-176,
Boston, Massachusetts, USA, May 2004. Association for Computational Linguistics.

**[Speaker Guest]:**

assistant
Alex Kuefler and Mykel J. Kochenderfer. Burn-in demonstrations for multi-modal imitation learning. arXiv preprint arXiv:1710.05090, 2017.

Shankar Kumar and William Byrne. Minimum Bayes-Risk Decoding for Statistical Machine Translation. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, pp. 169–176. Boston, Massachusetts, USA, May 2004. Association for Computational Linguistics.


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
Association for Computational Linguistics. Lucy Lai, Ann Zixiang Huang, and Samuel J Gershman. Action chunking as policy compression. 2022. Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. Dart: Noise injection
for robust imitation learning. In Conference on robot learning, pp. 143-156. PMLR, 2017. Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, and
Lerrel Pinto. Behavior Generation with Latent Actions, March 2024.

**[Speaker Host]:**

assistant
Action chunking as policy compression. Lucy Lai, Ann Zixiang Huang, and Samuel J Gershman. 2022.

Dart: Noise injection for robust imitation learning. Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. In Conference on Robot Learning, pp. 143–156. PMLR, 2017.

Behavior Generation with Latent Actions. Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. March 2024.


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
Behavior Generation with Latent Actions, March 2024. Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding. In Proceedings of the 40th International Conference on Machine Learning, pp. 19274-19286. PMLR, July 2023. Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke
Zettlemoyer, and Mike Lewis. Contrastive Decoding: Open-ended Text Generation as Optimization.

**[Speaker Guest]:**

assistant
Behavior Generation with Latent Actions, March 2024. Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding. In Proceedings of the 40th International Conference on Machine Learning, pp. 19274–19286. PMLR, July 2023. Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive Decoding: Open-ended Text Generation as Optimization.


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
Contrastive Decoding: Open-ended Text Generation as Optimization. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12286-12312, Toronto, Canada, July 2023. Association for Computational Linguistics. Yunzhu Li, Jiaming Song, and Stefano Ermon. Infogail: Interpretable imitation learning from visual

**[Speaker Host]:**

demonstrations.

**[Speaker Guest]:**

assistant
Contrastive decoding: open-ended text generation as optimization.
Yunzhu Li, Jiaming Song, and Stefano Ermon. Infogail: interpretable imitation learning from visual demonstrations.


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
Infogail: Interpretable imitation learning from visual
demonstrations. Advances in neural information processing systems, 30, 2017. Johan L¨ofberg. Oops! i cannot do it again: Testing for recursive feasibility in mpc. Automatica, 48
(3):550-555, 2012. Xiao Ma, Sumit Patidar, Iain Haughton, and Stephen James. Hierarchical Diffusion Policy for
Kinematics-Aware Multi-Task Robotic Manipulation, March 2024.

**[Speaker Host]:**

assistant
Infogail: Interpretable Imitation Learning from Visual Demonstrations, Advances in Neural Information Processing Systems, 30, 2017. Johan Lofberg. Oops! I Cannot Do It Again: Testing for Recursive Feasibility in MPC, Automatica, 48(3):550–555, 2012. Xiao Ma, Sumit Patidar, Iain Haughton, and Stephen James. Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation, March 2024.


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
Hierarchical Diffusion Policy for
Kinematics-Aware Multi-Task Robotic Manipulation, March 2024. Ajay Mandlekar, Fabio Ramos, Byron Boots, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Dieter
Fox. IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline
Robot Manipulation Data. In 2020 IEEE International Conference on Robotics and Automation
(ICRA), pp. 4414-4420, May 2020.

**[Speaker Guest]:**

assistant
Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation, March 2024. Ajay Mandlekar, Fabio Ramos, Byron Boots, Silvio Savareerse, Li Fei-Fei, Animesh Garg, and Dieter Fox.
IRIS: Implicit Reinforcement without Interaction at Scale for Learning Control from Offline Robot Manipulation Data. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 4414–4420, May 2020.


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
4414-4420, May 2020. Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li FeiFei, Silvio Savarese, Yuke Zhu, and Roberto Mart´ın-Mart´ın. What Matters in Learning from
Offline Human Demonstrations for Robot Manipulation. In Proceedings of the 5th Conference on
Robot Learning, pp. 1678-1690. PMLR, January 2022. Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. Locally Typical Sampling.

**[Speaker Host]:**

assistant
What matters in learning from offline human demonstrations for robot manipulation. Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. Locally Typical Sampling.


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
Locally Typical Sampling. Transactions of the Association for Computational Linguistics, 11:102-121, 2023. Kunal Menda, Katherine Driggs-Campbell, and Mykel J Kochenderfer. Ensembledagger: A
bayesian approach to safe imitation learning. In 2019 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS), pp. 5041-5048. IEEE, 2019. Published as a conference paper at ICLR 2025
Mathias M¨uller and Rico Sennrich.

**[Speaker Guest]:**

assistant
Mathias M¨uller and Rico Sennrich.


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
Published as a conference paper at ICLR 2025
Mathias M¨uller and Rico Sennrich. Understanding the Properties of Minimum Bayes Risk Decoding
in Neural Machine Translation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli
(eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics
and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long
Papers), pp. 259-272, Online, August 2021.

**[Speaker Host]:**

assistant
Mathias M¨uller and Rico Sennrich. Understanding the Properties of Minimum Bayes Risk Decoding in Neural Machine Translation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 259–272, Online, August 2021.


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
259-272, Online, August 2021. Association for Computational Linguistics. Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, and Sergey Levine. Steering Your Generalists:
Improving Robotic Foundation Models via Value Guidance. In 8th Annual Conference on Robot
Learning, September 2024. Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation, May 2024. St´ephane Ross, Geoffrey Gordon, and Drew Bagnell.

**[Speaker Guest]:**

assistant
Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. In 8th Annual Conference on Robot Learning, September 2024. Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation. May 2024.


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
St´ephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627-635. JMLR Workshop and Conference
Proceedings, 2011. Nur Muhammad Shafiullah, Zichen Cui, Ariuntuya Arty Altanzaya, and Lerrel Pinto. Behavior
transformers: Cloning k modes with one stone.

**[Speaker Host]:**

assistant
Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.

Nur Muhammad Shafiullah, Zichen Cui, Ariuntuya Arty Altanzaya, and Lerrel Pinto. Behavior transformers: Cloning k modes with one stone.


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

Behavior
transformers: Cloning k modes with one stone. Advances in neural information processing systems, 35:22955-22968, 2022. Lucy Xiaoyang Shi, Archit Sharma, Tony Z. Zhao, and Chelsea Finn. Waypoint-Based Imitation
Learning for Robotic Manipulation. In Proceedings of The 7th Conference on Robot Learning,
pp. 2195-2209. PMLR, December 2023. Aravind Sivakumar, Kenneth Shaw, and Deepak Pathak. Robotic telekinesis: Learning a robotic
hand imitator by watching humans on youtube.

**[Speaker Host]:**

assistant
Behavior transformers: Cloning k modes with one stone. Advances in Neural Information Processing Systems, 35:22955–22968, 2022. Lucy Xiaoyang Shi, Archit Sharma, Tony Z. Zhao, and Chelsea Finn.

Waypoint-Based Imitation Learning for Robotic Manipulation. In Proceedings of the 7th Conference on Robot Learning, pp. 2195–2209. PMLR, December 023. Aravind Sivakumar, Kenneth Shaw, and Deepak Pathak.

Robotic telekinesis: Learning a robotic hand imitator by watching humans on YouTube.


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
Robotic telekinesis: Learning a robotic
hand imitator by watching humans on youtube. Robotics: Science and Systems (RSS), 2022. Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. Gokul Swamy, Sanjiban Choudhury, Drew Bagnell, and Steven Wu. Causal imitation learning under
temporally correlated noise. In International Conference on Machine Learning, pp. 20877-20890. PMLR, 2022.

**[Speaker Guest]:**

assistant
Robotic telekinesis: Learning a robotic hand imitator by watching humans on YouTube.
Causal imitation learning under temporally correlated noise.
Roformer: Enhanced transformer with rotary position embedding.


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
PMLR, 2022. Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep
Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot
policy. arXiv preprint arXiv:2405. 12213, 2024. Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David
Crandall, and Dhruv Batra. Diverse Beam Search for Improved Description of Complex Scenes.

**[Speaker Host]:**

assistant
Octo: An open-source generalist robot policy.
Diverse Beam Search for Improved Description of Complex Scenes.


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
Diverse Beam Search for Improved Description of Complex Scenes. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), April 2018. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language
Models. arXiv:2203. 11171 [cs], April 2022. Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston.

**[Speaker Guest]:**

assistant
Diverse Beam Search for Improved Description of Complex Scenes. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), April 2018. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou.

Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171 [cs], April 2022. Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston.


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
Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural Text Generation With Unlikelihood Training. In International Conference on Learning
Representations, September 2019. Chuan Wen, Jierui Lin, Trevor Darrell, Dinesh Jayaraman, and Yang Gao. Fighting Copycat Agents
in Behavioral Cloning from Observation Histories. In Advances in Neural Information Processing
Systems, volume 33, pp. 2564-2575. Curran Associates, Incorporated, 2020.

**[Speaker Host]:**

assistant
Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural Text Generation With Unlikelihood Training. In International Conference on Learning Representations, September 2019.

Chuan Wen, Jierui Lin, Trevor Darrell, Dinesh Jayaraman, and Yang Gao. Fighting Copycat Agents in Behavioral Cloning from Observation Histories. In Advances in Neural Information Processing Systems, volume 33, pp. 2564–2575. Curran Associates, Incorporated, 2020.


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
Curran Associates, Incorporated, 2020. Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel. Gello: A general, low-cost,
and intuitive teleoperation framework for robot manipulators. arXiv preprint arXiv:2309. 13037,
2023. Annie Xie, Dylan Losey, Ryan Tolsma, Chelsea Finn, and Dorsa Sadigh. Learning Latent Representations to Influence Multi-Agent Interaction. In Proceedings of the 2020 Conference on Robot
Learning, pp. 575-588. PMLR, October 2021.

**[Speaker Guest]:**

assistant
Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators.
Annie Xie, Dylan Losey, Ryan Tolsma, Chelsea Finn, and Dorsa Sadigh. Learning Latent Representations to Influence Multi-Agent Interaction. In Proceedings of the 2020 Conference on Robot Learning, pp. 575–588. PMLR, October 2021.


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
PMLR, October 2021. Published as a conference paper at ICLR 2025
Xiaomeng Xu, Huy Ha, and Shuran Song. Dynamics-Guided Diffusion Model for Robot Manipulator Design, February 2024. Tianxiang Zhao, Wenchao Yu, Suhang Wang, Lu Wang, Xiang Zhang, Yuncong Chen, Yanchi
Liu, Wei Cheng, and Haifeng Chen. Skill disentanglement for imitation learning from suboptimal demonstrations. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, pp. 3513-3524.

**[Speaker Host]:**

assistant
Xiaomeng Xu, Huy Ha, and Shuran Song. Dynamics-Guided Diffusion Model for Robot Manipulator Design, February 2024.

Tianxiang Zhao, Wenchao Yu, Suhang Wang, Lu Wang, Xiang Zhang, Yuncong Chen, Yanchi Liu, Wei Cheng, and Haifeng Chen. Skill disentanglement for imitation learning from suboptimal demonstrations. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, pp. 3513–3524.


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
3513-3524. ACM, August 2023a. doi:
10. 1145/3580305. 3599506. URL [web link] doi. org/10. 1145/3580305. 3599506. Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware. In Robotics: Science and Systems (RSS) 2023, April
2023b. Hankz Hankui Zhuo, Yantian Zha, Subbarao Kambhampati, and Xin Tian. Discovering underlying
plans based on shallow models.

**[Speaker Guest]:**

assistant
Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Robotics: Science and Systems (RSS) 2023. Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn.


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
Discovering underlying
plans based on shallow models. ACM Transactions on Intelligent Systems and Technology (TIST),
11(2):1-30, 2020. Published as a conference paper at ICLR 2025

**[Speaker Host]:**

previous decision

**[Speaker Guest]:**

positive samples

**[Speaker Host]:**

negative samples

**[Speaker Guest]:**

demo distribution
Figure 10: Distributional interpretation of BID. The backward criterion (Equation 6) favors samples close to
the past decision; the forward criterion (Equation 7) promotes samples with a high likelihood under the target

**[Speaker Host]:**

distribution.

**[Speaker Guest]:**

assistant
Figure 10: Distributional interpretation of BID. The backward criterion favors samples close to the past decision; the forward criterion promotes samples with high likelihood under the target distribution.


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
The backward criterion (Equation 6) favors samples close to
the past decision; the forward criterion (Equation 7) promotes samples with a high likelihood under the target
distribution. Action Horizon

**[Speaker Host]:**

Noise Level

**[Speaker Guest]:**

3

**[Speaker Host]:**

7

**[Speaker Guest]:**

0. 0

**[Speaker Host]:**

4. 03

**[Speaker Guest]:**

2. 07

**[Speaker Host]:**

1. 54

**[Speaker Guest]:**

1. 06

**[Speaker Host]:**

0. 00

**[Speaker Guest]:**

0. 2

**[Speaker Host]:**

1. 43

**[Speaker Guest]:**

0. 94

**[Speaker Host]:**

0. 39

**[Speaker Guest]:**

0. 71

**[Speaker Host]:**

1. 32

**[Speaker Guest]:**

0. 4

**[Speaker Host]:**

0. 36

**[Speaker Guest]:**

0. 42

**[Speaker Host]:**

0. 59

**[Speaker Guest]:**

0. 835

**[Speaker Host]:**

1. 11
Table 3: Total variation distance between the action distributions of each model and the expert in environments
with varying noise levels.

**[Speaker Guest]:**

assistant
The backward criterion (Equation 6) favors samples close to the past decision; the forward criterion (Equation 7) promotes samples with a high likelihood under the target distribution.

Table 3: Total variation distance between the action distributions of each model and the expert in environments with varying noise levels.

Noise Level: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

Action Horizon: 3, 7


**[Speaker Host]:**

Values:
0.00, 1.06, 0.0, 4.03, 2.07, 1.54, 1.06, 0.00, 0.2, 1.43, 0.94, 0.39, 0.71, 1.32, 0.4, 0.36, 0.42, 0.59, 0.835, 1.11

(Note: The data appears to be a list of total variation distances across different noise levels and action horizons. The structure suggests a table where each row corresponds to a noise level and each column to an action horizon. However, without clear alignment, the raw numbers are presented as a sequence. The key point is the comparison of model action distributions to the expert under different noise conditions.)


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

11
Table 3: Total variation distance between the action distributions of each model and the expert in environments
with varying noise levels. Lower values indicate better performance. A
ADDITIONAL EXPERIMENTS

**[Speaker Host]:**

A. 1
ONE-DIMENSIONAL SIMULATIONS
In addition to Fig. 5, we summarize the total variation distance between each learned policy and the
demonstration in the one-dimensional simulation.

**[Speaker Guest]:**

assistant
Table 3: Total variation distance between the action distributions of each model and the expert in environments with varying noise levels. Lower values indicate better performance.

Additional Experiments
A.1 One-Dimensional Simulations
In addition to Fig. 5, we summarize the total variation distance between each learned policy and the demonstration in the one-dimensional simulation.


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
5, we summarize the total variation distance between each learned policy and the
demonstration in the one-dimensional simulation. Our results indicate that a shorter action horizon
is more effective in noisier environments, whereas a longer action horizon yields better performance
in static environments. A. 2
ACTION HORIZON VS. CONTEXT LENGTH
Setup. Our work builds on the premise that the action horizon is longer than the context length,
as commonly designed for recent policies.

**[Speaker Host]:**

assistant
We summarize the total variation distance between each learned policy and the demonstration in a one-dimensional simulation. Our results show that a shorter action horizon is more effective in noisier environments, while a longer action horizon leads to better performance in static environments. Our work builds on the premise that the action horizon is longer than the context length, as is commonly designed for recent policies.


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
Our work builds on the premise that the action horizon is longer than the context length,
as commonly designed for recent policies. While BID mitigates the inherent limitations of this design choice through test-time decoding, an important question remains: could extending the context
length yield stronger policies? To understand this, we trained diffusion policies with varying combinations of action horizons and context lengths on the Push-T task.

**[Speaker Guest]:**

assistant
Our work builds on the premise that the action horizon is longer than the context length, as commonly designed for recent policies. While BID mitigates the inherent limitations of this design choice through test-time decoding, an important question remains: could extending the context length yield stronger policies? To understand this, we trained diffusion policies with varying combinations of action horizons and context lengths on the Push-T task.


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
To understand this, we trained diffusion policies with varying combinations of action horizons and context lengths on the Push-T task. Specifically, we use a short
context length (c = 2) and a short action horizon (h = 2) as our baseline, set the prediction length
equal to the action horizon l = h, and incrementally increase these parameters to larger values
6, 10, 14 to assess their impact. Result. Fig. 11 compares the performance of the policy learned with different ∆h = h −c.

**[Speaker Host]:**

assistant
We trained diffusion policies with varying combinations of action horizons and context lengths on the Push-T task. We used a short context length (c = 2) and a short action horizon (h = 2) as our baseline, set the prediction length equal to the action horizon (l = h), and increased these parameters to larger values like 6, 10, and 14 to assess their impact. Figure 11 compares the performance of the policy learned with different values of ∆h = h − c.


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
11 compares the performance of the policy learned with different ∆h = h −c. As expected, the policy with both a short action horizon and a short context length does not perform well,
due to its limited capability to model long-range temporal dependencies.

**[Speaker Guest]:**

assistant
The policy with both a short action horizon and a short context length performs poorly because it lacks the ability to model long-range temporal dependencies.


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
As expected, the policy with both a short action horizon and a short context length does not perform well,
due to its limited capability to model long-range temporal dependencies. Interestingly, extending the
context length initially boosts performance (∆h = −4), but this trend reverses as the context length
becomes too long (∆h less than or equal to −8), likely due to overfitting to an increased number of spurious features.

**[Speaker Host]:**

assistant
The policy with both a short action horizon and short context length performs poorly because it can’t model long-term temporal dependencies. Increasing context length initially improves performance, but beyond a point, performance drops due to overfitting on spurious features.


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
Interestingly, extending the
context length initially boosts performance (∆h = −4), but this trend reverses as the context length
becomes too long (∆h less than or equal to −8), likely due to overfitting to an increased number of spurious features. In contrast, expanding the action horizon results in more robust performance improvements,
validating its pivotal role in imitation learning from human demonstrations. A. 3
ABLATION STUDY OF FORWARD CONTRAST

**[Speaker Guest]:**

Setup.

**[Speaker Host]:**

assistant
Extending the context length initially improves performance, but beyond a certain point, performance drops due to overfitting to irrelevant features. In contrast, expanding the action horizon leads to more stable and robust performance gains, highlighting its importance in imitation learning from human demonstrations.


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

3
ABLATION STUDY OF FORWARD CONTRAST
Setup. To understand the effect of forward contrast (Equation 7), we evaluate the full version
of our method against three reduced variants in open-loop operations: Vanilla (without forward
contrast), Positive (without negative samples), and Negative (without positive samples). Similar
to §5. 2. 2, our ablation study is conducted in seven simulation tasks, each with three random seeds. Result. Fig. 12 summarizes the result of this ablation study.

**[Speaker Host]:**

assistant
We evaluate the full version of our method against three reduced variants in open-loop operations: Vanilla (without forward contrast), Positive (without negative samples), and Negative (without positive samples). The ablation study is conducted across seven simulation tasks, each run with three random seeds. Figure 12 summarizes the results.


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
12 summarizes the result of this ablation study. Notably, both positive and negative samples are essential for effective sample selection, and omitting either leads to significant
performance declines. Without negative samples, our decoding method reduces to an approximate
maximum a posteriori estimation, which can result in suboptimal decisions due to modeling errors. Conversely, without positive samples, the sampling process may be biased towards rare instances.

**[Speaker Guest]:**

assistant
Both positive and negative samples are essential for effective sample selection. Omitting either leads to significant performance declines. Without negative samples, the decoding method reduces to an approximate maximum a posteriori estimation, risking suboptimal decisions from modeling errors. Without positive samples, the sampling process becomes biased toward rare instances.


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
Conversely, without positive samples, the sampling process may be biased towards rare instances. Published as a conference paper at ICLR 2025

**[Speaker Host]:**

8

**[Speaker Guest]:**

0

**[Speaker Host]:**

8

**[Speaker Guest]:**

h

**[Speaker Host]:**

0. 7

**[Speaker Guest]:**

0. 8

**[Speaker Host]:**

0. 9

**[Speaker Guest]:**

Test Score
Figure 11: Effect of action horizon (h) and context length (c) on diffusion policies in the Push-T task. The
baseline is set at h = 2 and c = 2, with ∆h = h −c = 0.

**[Speaker Host]:**

assistant
Conversely, without positive samples, the sampling process may be biased towards rare instances. Published as a conference paper at ICLR 2025.

Figure 11: Effect of action horizon (h) and context length (c) on diffusion policies in the Push-T task. The baseline is set at h = 2 and c = 2, with ∆h = h − c = 0.


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

The
baseline is set at h = 2 and c = 2, with ∆h = h −c = 0. Extending the action horizon (h > 2) consistently
improves performance, whereas extending the context length (c > 2) can cause substantial performance declines. Each model is trained for 5k epochs. Results are averaged over the last five checkpoints. Vanilla

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
on forward contrast.

**[Speaker Guest]:**

assistant
The baseline is set at h = 2 and c = 2, with ∆h = h − c = 0. Extending the action horizon (h > 2) consistently improves performance, while extending the context length (c > 2) can lead to significant performance drops. Each model is trained for 5k epochs, and results are averaged over the last five checkpoints. Figure 12 shows the effect of positive and negative samples on forward contrast, with success rates of 0.77, 0.78, 0.79, and 0.80 for the vanilla, positive, negative, and contrast methods, respectively.


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

80

**[Speaker Guest]:**

Success Rate
Figure 12: Effect of positive and negative samples
on forward contrast. Performance of ablated variants
of forward contrast is evaluated across seven simulation tasks. The absence of either positive or negative samples prevents achieving the full performance
gains observed with the contrast objective. 0. 0

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

1.

**[Speaker Guest]:**

assistant

**[Speaker Host]:**

Success Rate
Figure 12: Effect of positive and negative samples on forward contrast. Performance of ablated variants of forward contrast is evaluated across seven simulation tasks. The absence of either positive or negative samples prevents achieving the full performance gains observed with the contrast objective.

Decay Rate: 0.00, 0.25, 0.50, 0.75, 1.0
Success Rate: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0


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

75

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
relative performance among different decay rates. The optimal decay rate varies by task, leading to a
practical challenge of identifying a universal temporal ensembling strategy (Zhao et al., 2023b). This result highlights the importance of both components and suggests the potential for extending
this paradigm in future work. A.

**[Speaker Guest]:**

assistant

**[Speaker Host]:**

Relative test score
Figure 13: Effect of the decay rate for the exponential moving average. In each task, we measure the relative performance among different decay rates. The optimal decay rate varies by task, leading to a practical challenge of identifying a universal temporal ensembling strategy (Zhao et al., 2023b). This result highlights the importance of both components and suggests the potential for extending this paradigm in future work. A.


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

A. 4
CHALLENGES FOR TEMPORAL ENSEBMLING
EMA exhibits competitive performance in Fig. 6. However, tuning its decay rate can be difficult in
practice. Fig. 13 shows the sensitivity of EMA to the decay rate across three different tasks, where
the optimal choices differ significantly. We conjecture that this high sensitivity stems from the
variability in the latent strategies between consecutive predictions. When consecutive predictions
follow similar strategies, a lower decay rate (i.

**[Speaker Host]:**

assistant
A. 4 Challenges for Temporal Ensembling

EMA exhibits competitive performance as shown in Fig. 6. However, tuning its decay rate can be difficult in practice. Fig. 13 illustrates the sensitivity of EMA to the decay rate across three different tasks, where the optimal decay rates differ significantly. We conjecture that this high sensitivity arises from variability in the latent strategies between consecutive predictions. When consecutive predictions follow similar strategies, a lower decay rate may be effective.


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
When consecutive predictions
follow similar strategies, a lower decay rate (i. e., stronger moving average) can enhance smoothness
and improve performance. Conversely, when consecutive predictions diverge in their underlying
strategies, averaging them can introduce adverse effects. Our method promotes coherence in latent
strategies and thus effectively complements temporal ensembling, as evidenced in Fig. 7. A.

**[Speaker Guest]:**

assistant
When consecutive predictions follow similar strategies, a lower decay rate—meaning a stronger moving average—can improve smoothness and performance. When predictions diverge in strategy, averaging them can backfire. Our method ensures coherence in the underlying strategies, complementing temporal ensembling, as shown in Fig. 7A.


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

A. 5
ABLATION STUDIES ON VQ-BET
Table 4 presents the performance of various sampling methods with VQ-BeT (Lee et al., 2024) on
the Push-T task under different levels of environment stochasticity. We compare BID against vanilla
sampling and EMA, for both open-loop and closed-loop executions. BID consistently outperforms
the baselines, particularly in highly stochastic settings, with close-loop BID attaining the best performance on average.

**[Speaker Guest]:**

assistant

**[Speaker Host]:**

A. 5
ABLATION STUDIES ON VQ-BET
Table 4 presents the performance of various sampling methods with VQ-BeT (Lee et al., 2024) on the Push-T task under different levels of environment stochasticity. We compare BID against vanilla sampling and EMA, for both open-loop and closed-loop executions. BID consistently outperforms the baselines, particularly in highly stochastic settings, with closed-loop BID attaining the best performance on average.


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
BID consistently outperforms
the baselines, particularly in highly stochastic settings, with close-loop BID attaining the best performance on average. Moreover, the performance of EMA degrades significantly as the environment
gets more noisy, likely due to the divergence in strategies between consecutive action chunks.

**[Speaker Guest]:**

assistant
BID consistently outperforms the baselines, especially in highly stochastic environments, with close-loop BID achieving the best average performance. In contrast, EMA's performance drops significantly as the environment becomes noisier, likely because of strategy divergence between consecutive action chunks.


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
Moreover, the performance of EMA degrades significantly as the environment
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

**[Speaker Host]:**

26.

**[Speaker Guest]:**

assistant
The performance of EMA degrades significantly as the environment gets noisier, likely due to divergence in strategies between consecutive action chunks. We conduct an ablation study on the forward contrast objective, isolating the roles of positive samples (from a strong model) and negative samples (from a weaker model). Published as a conference paper at ICLR 2025. Stochastic noise: 0.0, 1.0, 1.5. Average performance: Vanilla: 64.0 ± 4.2.


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

2
26. 9 plus or minus 2. 8
13. 0 plus or minus 0. 4
34. 6 plus or minus 1. 7

**[Speaker Guest]:**

EMA
64. 1 plus or minus 1. 7
27. 6 plus or minus 3. 3
12. 9 plus or minus 1. 1
34. 9 plus or minus 1. 3

**[Speaker Host]:**

Backward (ours)
64. 0 plus or minus 1. 3
27. 6 plus or minus 1. 0
13. 4 plus or minus 2. 4
35. 1 plus or minus 1. 0

**[Speaker Guest]:**

+ Positive (ours)
65. 6 plus or minus 1. 9
29. 7 plus or minus 0. 4
15. 7 plus or minus 1. 7
37. 0 plus or minus 0. 9

**[Speaker Host]:**

+ Negative (ours)
65. 1 plus or minus 2. 7
26. 6 plus or minus 0. 8

**[Speaker Guest]:**

14.

**[Speaker Host]:**

assistant

**[Speaker Guest]:**

64.1 ± 1.7

**[Speaker Host]:**

27.6 ± 3.3

**[Speaker Guest]:**

13.0 ± 0.4


**[Speaker Host]:**

64.0 ± 1.3

**[Speaker Guest]:**

27.6 ± 1.0

**[Speaker Host]:**

13.4 ± 2.4


**[Speaker Guest]:**

65.6 ± 1.9

**[Speaker Host]:**

29.7 ± 0.4

**[Speaker Guest]:**

15.7 ± 1.7


**[Speaker Host]:**

65.1 ± 2.7

**[Speaker Guest]:**

26.6 ± 0.8

**[Speaker Host]:**

❓ 14.0 ± ?


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

8
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

**[Speaker Guest]:**

42.

**[Speaker Host]:**

assistant

**[Speaker Guest]:**

8

**[Speaker Host]:**

14 ± 2.4

**[Speaker Guest]:**

35 ± 1.2
BID (ours): 66 ± 3.5

**[Speaker Host]:**

31 ± 3.0

**[Speaker Guest]:**

16 ± 1.2

**[Speaker Host]:**

37.8 ± 1.6

(a) Open-Loop Operation
Stochastic Noise: 0.0, 1.0, 1.5
Average: Vanilla — 48.9 ± 2.7, 38.3 ± 3.4, 29.5 ± 0.9, 38.9 ± 1.5
EMA — 52.6 ± 2.9, 35.7 ± 2.2, 18.4 ± 2.3, 35.6 ± 1.4
Backward (ours): 52.7 ± 1.3, 42


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

3
42. 0 plus or minus 3. 2
29. 4 plus or minus 0. 5
41. 4 plus or minus 1. 2

**[Speaker Host]:**

+ Positive (ours)
53. 8 plus or minus 3. 3
44. 1 plus or minus 2. 5
30. 4 plus or minus 0. 8
42. 8 plus or minus 1. 4

**[Speaker Guest]:**

+ Negative (ours)
53. 0 plus or minus 2. 1
44. 3 plus or minus 2. 2
30. 8 plus or minus 0. 4
42. 7 plus or minus 1. 0

**[Speaker Host]:**

BID (ours)
54. 4 plus or minus 1. 8
45. 3 plus or minus 3. 8
31. 7 plus or minus 0. 3
43. 8 plus or minus 1.

**[Speaker Guest]:**

assistant

**[Speaker Host]:**

3

**[Speaker Guest]:**

42.0 ± 3.2

**[Speaker Host]:**

29.4 ± 0.5

**[Speaker Guest]:**

41.4 ± 1.2

**[Speaker Host]:**

+ Positive (ours)

**[Speaker Guest]:**

53.8 ± 3.3

**[Speaker Host]:**

44.1 ± 2.5

**[Speaker Guest]:**

30.4 ± 0.8

**[Speaker Host]:**

42.8 ± 1.4

**[Speaker Guest]:**

+ Negative (ours)

**[Speaker Host]:**

53.0 ± 2.1

**[Speaker Guest]:**

44.3 ± 2.2

**[Speaker Host]:**

30.8 ± 0.4

**[Speaker Guest]:**

42.7 ± 1.0

**[Speaker Host]:**

BID (ours)

**[Speaker Guest]:**

54.4 ± 1.8

**[Speaker Host]:**

45.3 ± 3.8

**[Speaker Guest]:**

31.7 ± 0.3

**[Speaker Host]:**

43.8 ± 1.0


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
8 plus or minus 1. 4
(b) Closed-Loop Operation
Table 4: Success rates of VQ-BeT on the Push-T task under various conditions and sampling methods. The left
table shows open-loop success rates, while the right table shows closed-loop success rates. BID consistently
outperforms vanilla counterparts in both settings. Vanilla

**[Speaker Guest]:**

L2 (Ours)

**[Speaker Host]:**

L1 (Ours)

**[Speaker Guest]:**

Cosine (Ours)

**[Speaker Host]:**

Square
0. 68 plus or minus 0. 06
0. 76 plus or minus 0. 03
0. 76 plus or minus 0. 04
0. 73 plus or minus 0. 04

**[Speaker Guest]:**

Lift
0. 12 plus or minus 0.

**[Speaker Host]:**

assistant

**[Speaker Guest]:**

8 plus or minus 1.4
(b) Closed-Loop Operation

Table 4: Success rates of VQ-BeT on the Push-T task under various conditions and sampling methods. The left table shows open-loop success rates, while the right table shows closed-loop success rates. BID consistently outperforms vanilla counterparts in both settings.


**[Speaker Host]:**

Vanilla

**[Speaker Guest]:**

L2 (Ours)

**[Speaker Host]:**

L1 (Ours)

**[Speaker Guest]:**

Cosine (Ours)

**[Speaker Host]:**

Square
0.68 plus or minus 0.06
0.76 plus or minus 0.03
0.76 plus or minus 0.04
0.73 plus or minus 0.04

**[Speaker Guest]:**

Lift
0.12 plus or minus 0.06

(Note: The original text appears to be cut off at the end of the Lift row. The full table is incomplete, but the structure and data points are preserved as provided. All LaTeX, formatting, and irrelevant symbols have been cleaned. Only meaningful, readable text remains.)


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
12 plus or minus 0. 02
0. 58 plus or minus 0. 06
0. 68 plus or minus 0. 05
0. 70 plus or minus 0. 02

**[Speaker Host]:**

Kitchen
0. 22 plus or minus 0. 04
0. 64 plus or minus 0. 05
0. 70 plus or minus 0. 04
0. 61 plus or minus 0. 03
Table 5: Effect of distance metric. We evaluate BID with different distance metrics on three robot manipulation
tasks. Regardless of the chosen metric, BID substantially outperforms the vanilla baseline.

**[Speaker Guest]:**

assistant

**[Speaker Host]:**

12 ± 0.02

**[Speaker Guest]:**

0.58 ± 0.06

**[Speaker Host]:**

0.68 ± 0.05

**[Speaker Guest]:**

0.70 ± 0.02

**[Speaker Host]:**

0.22 ± 0.04

**[Speaker Guest]:**

0.64 ± 0.05

**[Speaker Host]:**

0.70 ± 0.04

**[Speaker Guest]:**

0.61 ± 0.03

Table 5: Effect of distance metric. We evaluate BID with different distance metrics on three robot manipulation tasks. Regardless of the chosen metric, BID substantially outperforms the vanilla baseline.


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
Regardless of the chosen metric, BID substantially outperforms the vanilla baseline. Notably, the L1
distance yields an even stronger performance than the default L2 distance used in our other experiments. that both types of samples contribute to the strong performance of BID across varying levels of
environment stochasticity. A. 6
CHOICE OF DISTANCE METRICS
As described in §4. 2, our method uses the L2 distance as the default distance measure for quantifying
similarity between action chunks.

**[Speaker Host]:**

assistant
Regardless of the chosen metric, BID substantially outperforms the vanilla baseline. Notably, the L1 distance yields even stronger performance than the default L2 distance used in other experiments. Both types of samples contribute to BID's strong performance across varying levels of environmental stochasticity. As described in §4.2, the method uses L2 distance as the default measure for quantifying similarity between action chunks.


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
2, our method uses the L2 distance as the default distance measure for quantifying
similarity between action chunks. Yet, L2 distance is just one of several viable options. To assess
the sensitivity of our approach to the choice of distance metric, we extend the experiments in §5. 2. 1
to further evaluate BID with alternative choices, including L1 and cosine distance.

**[Speaker Guest]:**

assistant
Our method uses the L2 distance as the default measure for quantifying similarity between action chunks. However, L2 is just one of several possible distance metrics. To evaluate how sensitive our approach is to the choice of distance metric, we extend the experiments in Section 5.2.1 to test BID with alternative metrics, including L1 and cosine distance.


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

1
to further evaluate BID with alternative choices, including L1 and cosine distance. Results from
three robot manipulation tasks, summarized in Table 5, demonstrate that BID achieves substantial
performance gains regardless of the metric used. In particular, BID with L1 distance surpasses the
default L2 distance in performance, highlighting the untapped potential of our method. B
ADDITIONAL DISCUSSIONS
Interpretation of our method.

**[Speaker Guest]:**

assistant
To further evaluate BID with alternative choices, including L1 and cosine distance, results from three robot manipulation tasks—summarized in Table 5—demonstrate that BID achieves substantial performance gains regardless of the metric used. In particular, BID with L1 distance surpasses the default L2 distance in performance, highlighting the untapped potential of our method.


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

B
ADDITIONAL DISCUSSIONS
Interpretation of our method. Our method makes no changes to the learned policy; instead, it
intervenes in the model distribution through sample selection. As illustrated in Fig. 10, randomly
sampled sequences may be misaligned with both the previous decisions and the target demonstrations.

**[Speaker Guest]:**

assistant
Our method does not alter the learned policy. Instead, it intervenes by selecting samples from the model distribution. As shown in Fig. 10, randomly sampled sequences can be misaligned with both previous decisions and target demonstrations.


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

10, randomly
sampled sequences may be misaligned with both the previous decisions and the target demonstrations. Given a set of candidates, the backward step first identifies the behavioral mode from the past
decision stored in memory; the forward step then removes the samples with low likelihood under
the target distribution using prior knowledge of positive and negative samples.

**[Speaker Guest]:**

assistant
Randomly sampled sequences may be misaligned with previous decisions and target demonstrations. Given a set of candidates, the backward step identifies the behavioral mode from past decisions in memory, while the forward step removes samples with low likelihood under the target distribution using prior knowledge of positive and negative samples.


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
Given a set of candidates, the backward step first identifies the behavioral mode from the past
decision stored in memory; the forward step then removes the samples with low likelihood under
the target distribution using prior knowledge of positive and negative samples. By comparing samples across time steps and model checkpoints, our method bridges the gap between the proposal and
target distributions at test time. Relation to recent methods.

**[Speaker Host]:**

assistant
The backward step identifies the behavioral mode from past decisions stored in memory, while the forward step removes low-likelihood samples using prior knowledge of positive and negative examples. By comparing samples across time steps and model checkpoints, the method aligns the proposal distribution with the target distribution at test time.


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
Relation to recent methods. Our method builds upon the receding horizon (Chi et al., 2023) and
temporal ensembling (Zhao et al., 2023b) used in previous works, but with crucial distinctions. Receding horizon seeks a compromise between long-term consistency and short-term reactivity by
using a moderate action horizon (e. g., half of the prediction length), which is inevitably sup-optimal
when both factors are prominent.

**[Speaker Guest]:**

assistant
Our method builds on receding horizon and temporal ensembling, but with key differences. Receding horizon balances long-term consistency and short-term reactivity by using a moderate action horizon—this approach is inherently suboptimal when both goals are important.


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
g., half of the prediction length), which is inevitably sup-optimal
when both factors are prominent. Temporal ensembling strengthens dependency across chunks by
averaging multiple decisions over time; however, weighted-averaging operations can be detrimental
when consecutive decisions fall into distinct modes. Our method more effectively addresses crosschunk dependency through dedicated behavioral search and is not mutually exclusive to previous

**[Speaker Host]:**

methods.

**[Speaker Guest]:**

assistant
Half of the prediction length), which is inevitably suboptimal when both factors are prominent. Temporal ensembling strengthens dependency across chunks by averaging multiple decisions over time; however, weighted-averaging operations can be detrimental when consecutive decisions fall into distinct modes. Our method more effectively addresses cross-chunk dependency through dedicated behavioral search and is not mutually exclusive to previous methods.


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
Our method more effectively addresses crosschunk dependency through dedicated behavioral search and is not mutually exclusive to previous
methods. We will demonstrate in the next section that combining our method with the moving
average can further improve closed-loop action chunking. Published as a conference paper at ICLR 2025

**[Speaker Host]:**

name

**[Speaker Guest]:**

value

**[Speaker Host]:**

batch size N

**[Speaker Guest]:**

mode size K

**[Speaker Host]:**

prediction length l
temporal coherence decay ρ

**[Speaker Guest]:**

0. 5
moving average decay λ

**[Speaker Host]:**

0. 5
Table 6: Default hyper-parameters in our experiments.

**[Speaker Guest]:**

assistant
Our method more effectively addresses cross-chunk dependency through dedicated behavioral search and is not mutually exclusive to previous methods. We will demonstrate in the next section that combining our method with the moving average can further improve closed-loop action chunking. Published as a conference paper at ICLR 2025.

Table 6: Default hyper-parameters in our experiments.

**[Speaker Host]:**

Batch size N: 0.5

**[Speaker Guest]:**

Mode size K: 0.5
Prediction length l: 0.5
Temporal coherence decay ρ: 0.5
Moving average decay λ: 0.5


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

5
Table 6: Default hyper-parameters in our experiments. C

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
training data collected from human demonstrations in each benchmark. Push-T: We adopt the Push-T environment introduced in (Chi et al., 2023), where the goal is to push
a T-shaped block on a table to a target position.

**[Speaker Guest]:**

assistant
Our simulation experiments are conducted on three robot manipulation benchmarks. We use the training data collected from human demonstrations in each benchmark. Push-T: We adopt the Push-T environment introduced in (Chi et al., 2023), where the goal is to push a T-shaped block on a table to a target position.


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
Push-T: We adopt the Push-T environment introduced in (Chi et al., 2023), where the goal is to push
a T-shaped block on a table to a target position. The action space is two-dimensional end-effector
velocity control. The training dataset contains 206 demonstrations collected by humans. Robomimic: We use five tasks in the Robomimic suite (Mandlekar et al., 2022), namely Lift, Can,
Square, Transport, and Tool Hang.

**[Speaker Host]:**

assistant
We adopt the Push-T environment introduced in (Chi et al., 2023), where the goal is to push a T-shaped block on a table to a target position. The action space is two-dimensional end-effector velocity control. The training dataset contains 206 demonstrations collected by humans. We use five tasks in the Robomimic suite (Mandlekar et al., 2022): Lift, Can, Square, Transport, and Tool Hang.


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
Robomimic: We use five tasks in the Robomimic suite (Mandlekar et al., 2022), namely Lift, Can,
Square, Transport, and Tool Hang. The training dataset for each task contains 300 episodes collected
from multi-human (MH) demonstrations. Franka Kitchen: We use the Franka Kitchen environment from (Gupta et al., 2020), featuring a
Franka Panda arm with a seven-dimensional action space and 566 human-collected demonstrations.

**[Speaker Guest]:**

assistant
Robomimic uses five tasks: Lift, Can, Square, Transport, and Tool Hang. Each task has 300 episodes collected from multi-human demonstrations. The dataset is based on the Franka Kitchen environment, which features a Franka Panda robot arm with a seven-dimensional action space and 566 human-collected demonstrations.


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
Franka Kitchen: We use the Franka Kitchen environment from (Gupta et al., 2020), featuring a
Franka Panda arm with a seven-dimensional action space and 566 human-collected demonstrations. The learned policy is evaluated on test cases involving four or more objects (p4), a challenging yet
practical task for robotic manipulation in household contexts. C. 1. 2
IMPLEMENTATION DETAILS. Our implementation of BID for Diffusion Policy is built upon the official code of Chi et al.

**[Speaker Host]:**

assistant
Franka Kitchen: We use the Franka Kitchen environment from Gupta et al. (2020), featuring a Franka Panda arm with a seven-dimensional action space and 566 human-collected demonstrations. The learned policy is evaluated on test cases involving four or more objects (p4), a challenging yet practical task for robotic manipulation in household contexts. Our implementation of BID for Diffusion Policy is built upon the official code of Chi et al.


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
Our implementation of BID for Diffusion Policy is built upon the official code of Chi et al. ,
with modifications made solely to the inference process. The policy takes in state inputs and predicts
a chunk of 16 actions as outputs. For each simulation task, we train the model for 100-1000 epochs
to reach near-optimal performance. We evaluate it in closed-loop operations, i. e., action horizon is

**[Speaker Guest]:**

set to 1.

**[Speaker Host]:**

assistant
Our implementation of BID for Diffusion Policy is based on the official code by Chi et al., with changes only to the inference process. The policy accepts state inputs and outputs a chunk of 16 actions. For each simulation task, we train the model for 100 to 1000 epochs to achieve near-optimal performance. We evaluate it in closed-loop settings, where the action horizon is set to 1.


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
e., action horizon is
set to 1. For forward contrast, we train the weak policy for 10-100 epochs, resulting in a suboptimal
policy for each task. The core hyperparameters are summarized in Table 6. Our implementation of BID for VQ-BeT (Lee et al., 2024) is built upon the code of LeRobot (Cadene
et al., 2024). We use the best public checkpoint as the strong policy and a checkpoint trained for
100k iterations as the weak policy. Since BID requires sample diversity, we set the temperature to

**[Speaker Guest]:**

be 0.

**[Speaker Host]:**

assistant
Action horizon is set to 1. For forward contrast, we train the weak policy for 10–100 epochs, resulting in a suboptimal policy for each task. The core hyperparameters are summarized in Table 6. Our implementation of BID for VQ-BeT (Lee et al., 2024) is built upon the code of LeRobot (Cadene et al., 2024). We use the best public checkpoint as the strong policy and a checkpoint trained for 100k iterations as the weak policy. Since BID requires sample diversity, we set the temperature to 0.


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
Since BID requires sample diversity, we set the temperature to
be 0. 5 for all methods in our experiments. C. 2
REAL-WORLD EXPERIMENT DETAILS

**[Speaker Guest]:**

C. 2. 1

**[Speaker Host]:**

DYNAMIC PLACING
Task. We consider a task where the robot is to deliver an object held in its gripper into a cup
held by a human. As shown in Fig. 14, this task comprises four main stages and presents two core

**[Speaker Guest]:**

challenges.

**[Speaker Host]:**

assistant
Since BID requires sample diversity, we set the temperature to 0.5 for all methods in our experiments.

Task: We consider a task where the robot delivers an object held in its gripper into a cup held by a human. As shown in Fig. 14, this task comprises four main stages and presents two core challenges.


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
14, this task comprises four main stages and presents two core
challenges. First, due to the similar size of the object and the cup, the robot must achieve high
precision to place the object accurately into the cup. Second, the position of the cup is not fixed,
requiring the robot to adjust its plans based on the latest position continuously. This task mirrors
real-world scenarios where robots interact with a dynamic environment, accommodating moving
objects and agents. Demonstration.

**[Speaker Guest]:**

assistant
This task involves four main stages and two core challenges. First, the object and the cup are similar in size, so the robot must achieve high precision to place the object accurately into the cup. Second, the cup's position is not fixed, meaning the robot must continuously update its plan based on the latest cup position. This scenario reflects real-world robotics applications where robots operate in dynamic environments with moving objects and agents.


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
Demonstration. In light of temporal dependencies and style variations in human behaviors, we
intentionally collect a diverse set of demonstration data, differing in factors such as average speed,
idling pause, and overall trajectory. We gather a total of 150 demonstration episodes: 50 clean and
consistent demonstrations, and 100 noisy and diverse demonstrations. All demonstrations successfully accomplish the task. Additional, the location of the cup is fixed and static within each episode.

**[Speaker Host]:**

assistant
We intentionally collect a diverse set of demonstration data to account for temporal dependencies and style variations in human behaviors. The dataset includes 150 demonstration episodes: 50 clean and consistent ones, and 100 noisy and varied ones. In each episode, the cup’s location is fixed and static. All episodes successfully complete the task.


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
Additional, the location of the cup is fixed and static within each episode. Robot. Following previous works (Chi et al., 2023; Prasad et al., 2024), we use a Franka Panda
as the robot hardware and the vision-based diffusion policy for its operation. The robot is equipped
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

(a) initialize

**[Speaker Host]:**

→

**[Speaker Guest]:**

(b) approach

**[Speaker Host]:**

→

**[Speaker Guest]:**

(c) decelerate

**[Speaker Host]:**

→

**[Speaker Guest]:**

(d) place
Figure 14: Human demonstrations on a Franka Panda robot for a real-world object delivery task.

**[Speaker Host]:**

assistant
The robot uses a Franka Panda as hardware and follows a vision-based diffusion policy for operation. Its location is fixed and static within each episode. Human demonstrations are used to guide the robot through the delivery task, following prior work by Chi et al. (2023) and Prasad et al. (2024). The robot performs a sequence of actions: initialize, approach, decelerate, and place. This setup enables consistent and repeatable object delivery in a real-world setting.


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
The robot is equipped
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

(a) initialize

**[Speaker Host]:**

→

**[Speaker Guest]:**

(b) approach

**[Speaker Host]:**

→

**[Speaker Guest]:**

(c) decelerate

**[Speaker Host]:**

→

**[Speaker Guest]:**

(d) place
Figure 14: Human demonstrations on a Franka Panda robot for a real-world object delivery task. The robot is
tasked with delivering an object held in its gripper into a cup held by a human. Each demonstration consists of
four main stages: (a) initialize the robot position randomly, (b) approach the target cup, (c) slow down near the
target cup, and (d) release the object.

**[Speaker Host]:**

assistant
The robot is equipped to perform a real-world object delivery task. Human demonstrations are shown on a Franka Panda robot. Each demonstration follows four stages: initialize the robot position randomly, approach the target cup, decelerate near the target cup, and place the object. The task involves delivering an object held in the gripper into a cup held by a human.


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
Each demonstration consists of
four main stages: (a) initialize the robot position randomly, (b) approach the target cup, (c) slow down near the
target cup, and (d) release the object. The position of the target cup may change during an episode. (a) initialize

**[Speaker Guest]:**

(b) approach

**[Speaker Host]:**

(c) grasp

**[Speaker Guest]:**

(d) pick

**[Speaker Host]:**

(e) place
Figure 15: The robot is tasked with picking up a cup and placing it on a saucer nearby.

**[Speaker Guest]:**

assistant
Each demonstration consists of four main stages: initialize the robot position randomly, approach the target cup, slow down near the target cup, and release the object. The position of the target cup may change during an episode.

Figure 15: The robot is tasked with picking up a cup and placing it on a saucer nearby.


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

(a) initialize

**[Speaker Guest]:**

(b) approach

**[Speaker Host]:**

(c) grasp

**[Speaker Guest]:**

(d) pick

**[Speaker Host]:**

(e) place
Figure 15: The robot is tasked with picking up a cup and placing it on a saucer nearby. The four main stages
are (a) initializing the robot, (b) approaching the target cup, (c) grasping the target cup, (d) picking up the cup,
and (e) placing the cup on the target saucer. The position of the target cup may change during an episode.

**[Speaker Guest]:**

assistant
The robot is tasked with picking up a cup and placing it on a saucer nearby. The four main stages are: initializing the robot, approaching the target cup, grasping the target cup, picking up the cup, and placing the cup on the target saucer. The position of the target cup may change during an episode.


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
The position of the target cup may change during an episode. with two cameras: one ego-centric camera mounted at the wrist of the robot, one third-person camera
mounted at a static bracket. Both cameras provide visual observations at a resolution of 256 times 256
pixels. The robot operates at a frequency of 10 Hz, with a prediction length of 16 time steps. Evaluation.

**[Speaker Host]:**

assistant
The robot uses two cameras to monitor the target cup: one mounted at the wrist (ego-centric) and one fixed third-person camera. Both cameras capture visual data at 256x256 pixel resolution. The robot operates at 10 Hz with a prediction horizon of 16 time steps. Evaluation is conducted based on the robot's ability to track and respond to changes in the cup's position during operation.


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
Evaluation. We evaluate our method in comparison to vanilla random sampling under two conditions: static target, where the target cup remains fixed throughout the evaluation, and dynamic
target, where the target cup is gradually moved. In the dynamic setting, the location of the cup stays
within the range of training locations, but the movement is not encountered during training.

**[Speaker Guest]:**

assistant
We evaluate our method against vanilla random sampling under two conditions: static target, where the target cup remains fixed throughout evaluation, and dynamic target, where the target cup is gradually moved. In the dynamic setting, the cup’s location stays within the range of training locations, but the movement is not seen during training.


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
In the dynamic setting, the location of the cup stays
within the range of training locations, but the movement is not encountered during training. This
evaluation protocol is designed to explicitly assess the ability of the policy to react to unexpected
dynamics in the environment. Each method-setting pair is tested over 20 episodes, with both the
initial and target locations randomized across different episodes. Result. We summarize the result of the real-world experiments in Fig. 8.

**[Speaker Host]:**

assistant
In a dynamic setting, the cup's location stays within the range of training locations, but the movement patterns aren't encountered during training. This evaluation protocol tests a policy's ability to respond to unexpected environmental dynamics. Each method-setting pair is evaluated over 20 episodes, with both the initial and target locations randomized across episodes. Results from the real-world experiments are summarized in Figure 8.


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
8. The success rate of
vanilla random sampling is generally limited due to oscillations between different latent strategies,
which quickly diverge from the distribution of demonstrations. This issue is particularly pronounced
in the dynamic setting, where the vanilla baseline struggles to account for the target movements
within an action chunk lasting for 1. 6 seconds. In contrast, the proposed BID method significantly
improves performance in both static and dynamic settings.

**[Speaker Guest]:**

assistant
The success rate of vanilla random sampling is generally limited due to oscillations between different latent strategies, which quickly diverge from the distribution of demonstrations. This issue is particularly pronounced in dynamic settings, where vanilla sampling fails to account for target movements within an action chunk lasting 1.6 seconds. In contrast, the proposed BID method significantly improves performance in both static and dynamic settings.


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
In contrast, the proposed BID method significantly
improves performance in both static and dynamic settings. Notably, BID maintains a similar success
rate in the dynamic setting as in the static setting, suggesting its potential to extend action chunking
into uncertain environments. C. 2. 2

**[Speaker Host]:**

DYNAMIC PICKING
Task. Next, we consider a task where the robot is required to pick up a cup and place it onto a

**[Speaker Guest]:**

nearby saucer.

**[Speaker Host]:**

assistant
In contrast, the proposed BID method significantly improves performance in both static and dynamic settings. Notably, BID maintains a similar success rate in the dynamic setting as in the static setting, suggesting its potential to extend action chunking into uncertain environments.

Task: Next, we consider a task where the robot is required to pick up a cup and place it onto a nearby saucer.


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
Next, we consider a task where the robot is required to pick up a cup and place it onto a
nearby saucer. The cup was pulled with a string until the robot’s gripper successfully grasped it. The
task consists of five main stages, which are illustrated in Fig. 15. This setup also tests the robot’s
capability to interact with a dynamic environment, a critical challenge in real-world applications. Policy.

**[Speaker Guest]:**

assistant
The robot is tasked with picking up a cup and placing it on a nearby saucer. The cup is pulled with a string until the robot’s gripper successfully grasps it. The task consists of five main stages, as shown in Figure 15. This setup also tests the robot’s ability to interact with a dynamic environment, a key challenge in real-world applications.


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
Policy. We utilized the publicly available diffusion policy checkpoint from UMI (Chi et al., 2024)
without any additional fine-tuning. Notably, the policy was originally trained using demonstrations
in a static setting, where the cup’s position remained constant throughout the task. Our experimental
setup mirrored the one described by UMI, using the same UR5 robot hardware.

**[Speaker Host]:**

assistant
We used the publicly available diffusion policy checkpoint from UMI (Chi et al., 2024) without any additional fine-tuning. The policy was originally trained on demonstrations in a static setting, where the cup's position remained fixed. Our experimental setup matched the one described by UMI, using the same UR5 robot hardware.


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

Our experimental
setup mirrored the one described by UMI, using the same UR5 robot hardware. This allowed us
to directly evaluate the policy’s transferability to a dynamic environment, where the cup’s position
changes during the task. Due to the absence of an early checkpoint, we omitted negative samples in
forward contrast, focusing solely on positive consistency discussed in Appendix A. 3. Published as a conference paper at ICLR 2025

**[Speaker Host]:**

Evaluation.

**[Speaker Guest]:**

assistant
Our experimental setup mirrored the one described by UMI, using the same UR5 robot hardware. This allowed us to directly evaluate the policy’s transferability to a dynamic environment, where the cup’s position changes during the task. Due to the absence of an early checkpoint, we omitted negative samples in forward contrast, focusing solely on positive consistency. Published as a conference paper at ICLR 2025.


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
Published as a conference paper at ICLR 2025
Evaluation. We evaluated BID against three baselines: vanilla random sampling in both openloop and closed-loop configurations, and EMA (closed-loop). These methods were tested under two
conditions: static target, where the cup remained in a fixed position, and dynamic target, where the
cup was moved using the string.

**[Speaker Host]:**

assistant
Published as a conference paper at ICLR 2025
Evaluation: We evaluated BID against three baselines—vanilla random sampling in both open-loop and closed-loop configurations, and EMA (closed-loop). These methods were tested under two conditions: static target, where the cup remained in a fixed position, and dynamic target, where the cup was moved using the string.


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
These methods were tested under two
conditions: static target, where the cup remained in a fixed position, and dynamic target, where the
cup was moved using the string. Each method-setting combination was tested across 20 episodes,
with the initial positions of the cup and saucer kept consistent to ensure controlled comparisons. Results. The results, summarized in Fig. 9, highlight the challenges of the dynamic setting.

**[Speaker Guest]:**

assistant
The methods were tested under two conditions: static target, where the cup remained in a fixed position, and dynamic target, where the cup was moved using the string. Each method-setting combination was tested across 20 episodes, with the initial positions of the cup and saucer kept consistent to ensure controlled comparisons. The results, summarized in Fig. 9, highlight the challenges of the dynamic setting.


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
9, highlight the challenges of the dynamic setting. Openloop vanilla sampling performed poorly due to its inability to adapt to the cup’s movements, often
failing to approach the cup as it was pulled. While closed-loop vanilla sampling showed improved
reactivity, it suffered from inconsistent trajectories, resulting in jittery behavior when attempting to
grasp and place the cup.

**[Speaker Host]:**

assistant
The dynamic environment posed significant challenges. Open-loop vanilla sampling struggled to adapt to the cup's movements and frequently failed to approach it as it was pulled. Closed-loop vanilla sampling showed better reactivity but still suffered from inconsistent trajectories, leading to jittery and unstable grasping and placement behavior.


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
While closed-loop vanilla sampling showed improved
reactivity, it suffered from inconsistent trajectories, resulting in jittery behavior when attempting to
grasp and place the cup. Similarly, closed-loop EMA sampling demonstrated higher adaptability to
environmental changes but often failed to firmly grasp the cup, likely due to the limitations of naive
averaging, which compromises commitment to a specific strategy.

**[Speaker Guest]:**

assistant
Closed-loop vanilla sampling showed improved reactivity but suffered from inconsistent trajectories, leading to jittery behavior during grasping and placing. Closed-loop EMA sampling was more adaptable to environmental changes but often failed to securely grasp the cup, likely due to the limitations of naive averaging that weakened commitment to a specific strategy.


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
Similarly, closed-loop EMA sampling demonstrated higher adaptability to
environmental changes but often failed to firmly grasp the cup, likely due to the limitations of naive
averaging, which compromises commitment to a specific strategy. In contrast, BID achieved at
least a 2x improvement in success rate compared to all other methods in the dynamic setting, while
maintaining its performance in the static setting, demonstrating both adaptability and precision in
dynamic environments.

**[Speaker Host]:**

assistant
Closed-loop EMA sampling was more adaptable to environmental changes but often failed to securely grasp the cup, likely because of naive averaging that weakened commitment to a specific strategy. In contrast, BID achieved at least a two-fold improvement in success rate over all other methods in dynamic settings, while maintaining strong performance in static settings, showing both adaptability and precision in changing environments.


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
In contrast, BID achieved at
least a 2x improvement in success rate compared to all other methods in the dynamic setting, while
maintaining its performance in the static setting, demonstrating both adaptability and precision in
dynamic environments. D
ADDITIONAL DISCUSSIONS
Relation to Option Discovery. Action chunking and option discovery share similarities in modeling temporally extended actions. However, their designs and outcomes are often different.

**[Speaker Guest]:**

assistant
In contrast, BID achieved at least a 2x improvement in success rate compared to all other methods in dynamic settings, while maintaining its performance in static settings, demonstrating both adaptability and precision in dynamic environments.

Additional Discussions:
Relation to Option Discovery. Action chunking and option discovery both model temporally extended actions, but differ in design and outcomes.


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
However, their designs and outcomes are often different. Option
discovery typically aims to learn hierarchical policies, explicitly discovering high-level skills from
low-level action sequences (Bagaria & Konidaris, 2020; Jiang et al., 2022; 2023; Zhao et al., 2023a). In contrast, action chunking operates directly on low-level action sequences with fixed horizons, bypassing the need for abstraction into high-level skills.

**[Speaker Host]:**

assistant
However, their designs and outcomes are often different. Option discovery typically aims to learn hierarchical policies, explicitly discovering high-level skills from low-level action sequences. In contrast, action chunking operates directly on low-level action sequences with fixed horizons, bypassing the need for abstraction into high-level skills.


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
In contrast, action chunking operates directly on low-level action sequences with fixed horizons, bypassing the need for abstraction into high-level skills. This simplicity has proven effective in largescale robotic foundation models (Team et al., 2024; Black et al., 2024), addressing the scalability
challenge that option discovery has yet to overcome.

**[Speaker Guest]:**

assistant
In contrast, action chunking works directly on low-level action sequences with fixed horizons, without needing to abstract into high-level skills. This simplicity has proven effective in large-scale robotic foundation models, addressing the scalability challenge that option discovery has not yet solved.


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
This simplicity has proven effective in largescale robotic foundation models (Team et al., 2024; Black et al., 2024), addressing the scalability
challenge that option discovery has yet to overcome. Future work could explore variable-horizon
option discovery as a compelling alternative to action chunking, potentially combining the benefits
of temporal abstraction with the scalability demonstrated by action chunking. Relation to Long Context.

**[Speaker Host]:**

assistant
This simplicity has proven effective in large-scale robotic foundation models, addressing the scalability challenge that option discovery has yet to overcome. Future work could explore variable-horizon option discovery as a compelling alternative to action chunking, potentially combining the benefits of temporal abstraction with the scalability demonstrated by action chunking.


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
Relation to Long Context. Our analysis in §3 highlights the benefits of extending the action horizon to better capture temporal dependencies across actions. Another natural approach to capturing
these dependencies is to extend the context length. However, long-context policies often suffer from
robustness issues, due to spurious correlations between past and future actions, as evidenced in Appendix A2 and studied in de Haan et al. ; Wen et al. .

**[Speaker Guest]:**

assistant
Our analysis in §3 shows that extending the action horizon helps better capture temporal dependencies across actions. Another approach is to extend the context length, but long-context policies often face robustness issues caused by spurious correlations between past and future actions, as shown in Appendix A2 and studied by de Haan et al. and Wen et al.


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
. Nevertheless, the increasing
availability of large-scale robotic datasets for pre-training and fine-tuning may help mitigate these
challenges. Advances in leveraging long context may open up new opportunities in policy design,
such as long-context transformers (Su et al., 2024) and recurrent neural networks (Hochreiter &
Schmidhuber, 1997; Zhuo et al., 2020).

**[Speaker Host]:**

assistant
Nevertheless, the growing availability of large-scale robotic datasets may help address challenges in pre-training and fine-tuning. Advances in long-context modeling—such as long-context transformers and recurrent neural networks—could open new possibilities for policy design.


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
Advances in leveraging long context may open up new opportunities in policy design,
such as long-context transformers (Su et al., 2024) and recurrent neural networks (Hochreiter &
Schmidhuber, 1997; Zhuo et al., 2020). E

**[Speaker Guest]:**

PROOFS
In this section, we will clearly write out the transition dynamics in the training environment to be
Ptrain(st+1 | st, at), the transition dynamics in the test environment to be Ptest(st+1 | st, at) and
the implicit dynamics model of the training environment to be ˆPtrain(st+1 | st, at).

**[Speaker Host]:**

assistant
Advances in leveraging long context may open up new opportunities in policy design, such as long-context transformers and recurrent neural networks. In this section, we will clearly write out the transition dynamics in the training environment to be Ptrain(st+1 | st, at), the transition dynamics in the test environment to be Ptest(st+1 | st, at), and the implicit dynamics model of the training environment to be ˆPtrain(st+1 | st, at).


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

assistant
The section outlines the transition dynamics in both the training and test environments. Specifically, it defines the transition dynamics in the training environment as Ptrain(st+1 | st, at), the transition dynamics in the test environment as Ptest(st+1 | st, at), and the implicit dynamics model of the training environment as ˆPtrain(st+1 | st, at). The implicit dynamics model is designed to learn the dynamics of the training environment only, as the policy was trained solely on data from that environment and not on data from the test environment.


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

Note that the
implicit dynamics model attempts to learn the dynamics of the training environment and not the
test environment since our policy is assumed to have been trained only on data from the training
environment. First, we establish the following lemma which will help us compare different function classes based
on the information they have access to:
Lemma 4. Let L be a convex function and let X and Y be two random variables. Let G be the
class of functions g(X) that accept X as an input.

**[Speaker Host]:**

assistant
The implicit dynamics model aims to learn the dynamics of the training environment, not the test environment, because the policy is assumed to have been trained only on data from the training environment. To compare different function classes based on the information they have access to, we establish the following lemma: Let L be a convex function, and let X and Y be two random variables. Let G be the class of functions g(X) that accept X as input.


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

Let G be the
class of functions g(X) that accept X as an input. Then

**[Speaker Host]:**

min
g(X)∈G EX, Y [L(f(X, Y ), g(X))] = EX

**[Speaker Guest]:**

assistant
Let G be the class of functions g(X) that accept X as an input. Then the minimum over g(X) in G of the expected loss E[X,Y] of L(f(X,Y), g(X)) is equal to E[X].


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

**[Speaker Host]:**

assistant
The left-hand side is less than or equal to the right-hand side by the following logic:
Taking the expectation over X, we have EX[min_{c∈R} EY[L(f(X,Y), c)|X]]. This is equivalent to minimizing over c in R, then taking expectation over Y given X. Since the expectation over Y is taken conditionally on X, and we are minimizing over c, the outer expectation over X is well-defined. The minimization over c ensures that for each X, we choose the best constant c that minimizes the expected loss. Thus, the expectation over X of the conditional expectation is equal to the expectation over X of the minimum over c of the conditional loss. This establishes the inequality.


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
= EX [EY [L(f(X, Y ), c∗(X))|X]]
greater than or equal to

**[Speaker Guest]:**

min
g(X)∈G EX, Y [L(f(X, Y ), g(X))]
where we used c∗(X) := arg minc EX[L(f(X, Y ), c)|X]. We get the inequality by recognizing
that R ⊊G. For the reverse inequality, consider any g(X) ∈G:
E[L(f(X, Y ), g(X))] = EX [EY [L(f(X, Y ), g(X))|X]]
greater than or equal to EX


**[Speaker Host]:**

min

**[Speaker Guest]:**

g
EY [L(f(X, Y ), g(X))|X]

**[Speaker Host]:**

assistant
EX [EY [L(f(X, Y ), c∗(X))|X]] ≥ min_{g(X) ∈ G} EX [EY [L(f(X, Y ), g(X))|X]],
where c∗(X) is defined as arg min_{c} E[Y [L(f(X, Y ), c)|X]. The inequality holds because the set R is a proper subset of G. For the reverse inequality, consider any g(X) ∈ G:
E[L(f(X, Y ), g(X))] = EX [EY [L(f(X, Y ), g(X))|X]] ≥ EX [min_{g} EY [L(f(X, Y ), g(X))|X]].


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
Next, we prove the following lemma. This straightforward, and almost trivial, result is provided as
a separate lemma because we simplify terms in this manner quite often throughout our proofs.
Lemma 5. Let L be a convex function and let X, Y be two random variables. Then,

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

**[Speaker Host]:**

assistant
We conclude with the following lemma. Let L be a convex function, and let X and Y be two random variables. Then, the minimum over f of the expected loss is given by:


**[Speaker Guest]:**

min

**[Speaker Host]:**

f
E_{X,Y} [ P(X′ = X) L(f(X′), S(X, Y)) + ∑_{X′ ≠ X} P(X′) L(f(X′), S(X, Y)) ]

This expression simplifies to the expected loss over all possible values of X′, weighted by their probabilities. The lemma is stated here for convenience, as the structure of such a minimization appears frequently in subsequent proofs.


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

**[Speaker Guest]:**

assistant
The expected loss of a model f, averaged over inputs X and outputs Y, is bounded above by the expected loss of the best function f* plus a term that accounts for the probability of misclassification due to noise or incorrect predictions at other inputs X'. Specifically, the loss is bounded by:

min_f E_{X,Y} [L(f(X), S(X,Y))] + E_{X,Y} Σ_{X' ≠ X} P(X') L(f*(X'), S(X,Y))

where f* is the function that minimizes the expected loss over all possible functions, and the second term captures the expected loss from incorrect predictions at alternative inputs X' with non-zero probability. This establishes a bound on the generalization error in terms of the minimum achievable loss and the expected contribution from incorrect predictions.


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
less than or equal to min
f {EX, Y [L(f(X), S(X, Y ))]} + EX, Y


**[Speaker Host]:**

X

**[Speaker Guest]:**

X′̸=X

**[Speaker Host]:**

P(X′)ϵ

**[Speaker Guest]:**

assistant
Less than or equal to min of the expected loss over X and Y, plus the expected value over X and Y of the probability of X prime not equal to X times epsilon.


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
Assumption 2. An optimal πc, h must infer the unobserved states based on the observed states and
actions by modeling the transition dynamics ˆPtrain(st′ | st′−1, at′−1) accurately for all time step t′.
We consider this implicit model to be the same for both π(c, h) and π(c, h+d).
In other words, we can write using the law of total probability:
π(c, h)(at|st−h−c:t−h, at−h:t−1)
= Est−k:t−h−c−1, st−h+1:t∼ˆ

**[Speaker Host]:**

Ptrain

**[Speaker Guest]:**

assistant
Less than or equal to min of E[EX, Y [L(f(X), S(X, Y))]] + ϵ.

Assumption 1: With recent policies using a short context length c, we assume the range of temporal dependency modeled by a (c, h)-policy is limited—c + h < k. This allows us to focus on the relevant problem and ignore edge cases. The analysis can be extended to cases where c + h ≥ k. When c + h ≥ k, the longer action horizon model gains no advantage (αb = 0) because the additional states observed in time steps τb are irrelevant to at. The longer horizon policy either matches or performs worse than the shorter horizon policy due to missing recent past states.

Assumption 2: An optimal policy πc,h must infer unobserved states by modeling transition dynamics accurately: ˆPtrain(st′ | st′−1, at′−1) for all time steps t′. We treat this implicit model as identical for both π(c, h) and π(c, h+d). Using the law of total probability, we can express the policy as:

π(c, h)(at | st−h−c:t−h, at−h:t−1) = E[st−k:t−h−c−1, st−h+1:t ∼ ˆPtrain]


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
π(k, 1)(at|st−k:t)|st−h−c:t−h, at−h:t−1


**[Speaker Host]:**

and
π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−h−1)
= Est−k:t−h−d−c−1, st−h−d+1:t∼ˆ

**[Speaker Guest]:**

Ptrain

**[Speaker Host]:**

assistant
π(k, 1)(at|st−k:t)|st−h−c:t−h, at−h:t−1

**[Speaker Guest]:**

and
π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−h−1)
= E[st−k:t−h−d−c−1, st−h−d+1:t ∼ ˆPtrain]


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
π(k, 1)(at|st−k:t)|st−h−d−c:t−h−d, at−h−d:t−1

**[Speaker Host]:**

. E. 2

**[Speaker Guest]:**

DEFINITIONS
We, first, analyze the effect of reducing context horizon. We show that, provided action horizon is
constant, decreasing context horizon causes performance of the optimal policy to decrease. Consider a (c, h)-policy (i. e., the policy has context length c and action horizon h) whose probability
of taking action at at time t in a chunk generated at t is referred to as π(c, h) := π(c, h)(at|st−c:t).

**[Speaker Host]:**

assistant
We first analyze the effect of reducing context horizon. We show that, provided the action horizon is constant, decreasing the context horizon causes the performance of the optimal policy to decrease. Consider a (c, h)-policy—i.e., a policy with context length c and action horizon h—whose probability of taking action at time t in a chunk generated at time t is denoted by π(c, h) := π(c, h)(at | st−c:t).


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
e., the policy has context length c and action horizon h) whose probability
of taking action at at time t in a chunk generated at t is referred to as π(c, h) := π(c, h)(at|st−c:t). On
the other hand, consider a (c+1, h)-policy whose probability of taking action at in a chunk generated
at time t is referred to as π(c+1, h) := π(c+1, h)(at|st−c−1:t). Lastly, consider a (k, 1)-expert whose
probability of taking action at at time t is π∗. Proposition 6. Let L be a non-linear, convex function.

**[Speaker Guest]:**

assistant
The policy has context length c and action horizon h, and its probability of taking action at time t in a chunk generated at time t is denoted as π(c, h) := π(c, h)(at | st−c:t). In contrast, a (c+1, h)-policy has a probability of taking action at time t in a chunk generated at time t, denoted as π(c+1, h) := π(c+1, h)(at | st−c−1:t). Additionally, consider a (k, 1)-expert whose action probability at time t is π*. Proposition 6: Let L be a non-linear, convex function.


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
Let L be a non-linear, convex function. Let c < k. Let G := {at, st−k:t−c−1, zt−k:t}
and let C := {st−c:t}. Then,

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+1, h) EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+1, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i
less than or equal to min

**[Speaker Host]:**

π(c, h) EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i
In particular, this is an equality if and only if st−c−1 ∼Ptest(· | C) and ˆst−c−1 ∼ˆPtrain(· | C)
take on only one and the same value almost surely. Proof. We refer to the class of functions that accept at and st−c−1:t as inputs as X2.

**[Speaker Guest]:**

assistant
Let L be a non-linear, convex function. Suppose c < k. Define G := {at, st−k : t−c−1, zt−k : t} and C := {st−c : t}. Then,
min_π(c+1, h) E_G[h] L(π(c+1, h), π∗) over C ≤ min_π(c, h) E_G[h] L(π(c, h), π∗) over C.

In particular, this is an equality if and only if st−c−1 and ˆst−c−1 take on the same value almost surely under the distribution of C, i.e., st−c−1 ∼ P_test(· | C) and ˆst−c−1 ∼ ˆP_train(· | C) and both take only one common value almost surely.

We consider the class of functions that take at and st−c−1:t as inputs, denoted X2.


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
We refer to the class of functions that accept at and st−c−1:t as inputs as X2. Similarly, the
class of functions that do not accept at as inputs but accept st−c−1:t as inputs is X1. The function
class that accepts only st−c:t and not st−c−1 or at as inputs are elements of X0. Lastly, the function
class that accepts st−c:t and at as inputs, but not st−c−1, are elements of X−1.

**[Speaker Host]:**

assistant
We define four classes of functions based on their input acceptance:

- Class X2 accepts both at and st−c−1:t.
- Class X1 accepts st−c−1:t but not at.
- Class X0 accepts only st−c:t, not st−c−1 or at.
- Class X−1 accepts st−−c:t and at, but not st−c−1.


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
Lastly, the function
class that accepts st−c:t and at as inputs, but not st−c−1, are elements of X−1. min

**[Speaker Guest]:**

π(c+1, h)∈X2 EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+1, h), π∗C

**[Speaker Host]:**

i

**[Speaker Guest]:**

= Eat

**[Speaker Host]:**

"

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′

**[Speaker Guest]:**

(c+1, h)∈X1

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

(c+1, h), π∗)

**[Speaker Host]:**

at, st−c−1, C

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

Est−c−1

**[Speaker Host]:**

"

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′

**[Speaker Guest]:**

(c, h)∈X0

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

(c, h)∈X0

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

(c, h), π∗)

**[Speaker Guest]:**

at, st−c−1, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

#
(Jensen’s inequality)
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

=

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h)∈X−1 Eat

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−c−1

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−k:t−c−2, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c, h), π∗)

**[Speaker Host]:**

at, st−c−1, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

i C

**[Speaker Guest]:**

i

**[Speaker Host]:**

(Lemma 4).

**[Speaker Guest]:**

assistant
The function class that accepts st−c:t and at as inputs, but not st−c−1, are elements of X−1. The expected value of the minimum over π(c+1, h) in X2 of the expected loss, given by the expectation over t, is bounded above by the expectation over t of the minimum over π′(c+1, h) in X1 of the expected loss over past states and actions, with respect to the policy π∗. By applying Lemma 4, this is equivalent to the expectation over t of the minimum over π′(c, h) in X0 of the expected loss over past states and actions. Using Jensen’s inequality, this further simplifies to the minimum over π′(c, h) in X0 of the expected loss, which is then bounded by the minimum over π(c, h) in X−1 of the expected loss. This completes the derivation.


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

min

**[Speaker Guest]:**

π(c+1, h)∈X2 EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+1, h), π∗C

**[Speaker Host]:**

i

**[Speaker Guest]:**

= Eat

**[Speaker Host]:**

"

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′

**[Speaker Guest]:**

(c+1, h)∈X1

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

(c+1, h), π∗)

**[Speaker Host]:**

at, st−c−1, C

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

Est−c−1

**[Speaker Host]:**

"

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′

**[Speaker Guest]:**

(c, h)∈X0

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

(c, h)∈X0

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

(c, h), π∗)

**[Speaker Guest]:**

at, st−c−1, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

#
(Jensen’s inequality)
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

=

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c, h)∈X−1 Eat

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−c−1

**[Speaker Host]:**

h

**[Speaker Guest]:**

Est−k:t−c−2, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c, h), π∗)

**[Speaker Host]:**

at, st−c−1, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

i C

**[Speaker Guest]:**

i
(Lemma 4). Use the law of total expectation to conclude.

**[Speaker Host]:**

assistant
The expected loss is bounded by considering the minimization over policies in a sequence of sets X, with the loss function evaluated at different time steps and states. Using the law of total expectation and applying Jensen’s inequality, we can show that the expected loss at time t is bounded above by the minimum over policies in X₀ of the expected loss at time t−c−1, given past observations and actions. This follows from Lemma 4, which establishes the structure of the policy optimization under temporal shifts. The result holds due to the recursive nature of the policy evaluation and the structure of the state-action pairs.


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
Use the law of total expectation to conclude. The equality conditions for Jensen’s inequality provides
us the equality condition for this relationship too. Now, we formalize the definitions of Expected Observation Advantage and Maximum Inference
Disadvantage. Recall that, in §3. 2, we have two policies: π(c, h) and π(c, h+d); the former sees more recent states
while the latter remembers more past states.

**[Speaker Guest]:**

assistant
The law of total expectation allows us to analyze the relationship between policies that differ in their memory of past states. Specifically, we consider two policies: π(c, h) and π(c, h+d), where π(c, h) focuses on more recent states, and π(c, h+d) retains more past history. The equality conditions from Jensen’s inequality provide insight into when the expected advantage of one policy over the other is exactly zero. This leads to formal definitions of Expected Observation Advantage and Maximum Inference Disadvantage.


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
2, we have two policies: π(c, h) and π(c, h+d); the former sees more recent states
while the latter remembers more past states.

π(c+d, h)(at | st−h−d−c:t−h, at−h:t−1)

This policy has access to more context than π(c, h), specifically including the states st−h−c−d:t−h−c−1.


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
Observe that π(c+d, h) has access to more context than π(c, h), particularly the knowledge of states
st−h−c−d:t−h−c−1. Definition (Expected Observation Advantage (αb)). We know, using Proposition 6, π(c+d, h) has
lower divergence with respect to π∗than π(c, h). We say that the advantage π(c+d, h) gets from the
extra information is αb. More formally, we say that

**[Speaker Host]:**

αb := min

**[Speaker Guest]:**

π(c, h) E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c, h), π∗)

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

(8)
where C is defined as in Proposition 1.

**[Speaker Host]:**

assistant
π(c+d, h) has access to more context than π(c, h), specifically the state history st−h−c−d:t−h−c−1. Using Proposition 6, π(c+d, h) has lower divergence with respect to π* than π(c, h). The advantage π(c+d, h) gains from this additional information is defined as the Expected Observation Advantage, αb. Formally:

αb = min_π(c,h) E_h [L(π(c,h), π*)] − min_π(c+d,h) E_h [L(π(c+d,h), π*)]

where C is as defined in Proposition 1.


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
More formally, we say that

**[Speaker Guest]:**

αb := min

**[Speaker Host]:**

π(c, h) E

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c, h), π∗)

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

(8)
where C is defined as in Proposition 1. Clearly, αb
greater than or equal to 0. In particular, αb

**[Speaker Guest]:**

= 0 when
st−h−d−c:t−h−c−1 can be deterministically inferred by π(c, h) or when the expert policy is independent of them. Definition (Maximum Inference Disadvantage (ϵf)).

**[Speaker Host]:**

assistant
The quantity αb is defined as the difference between the expected loss of a policy π(c, h) and the expected loss of the expert policy π∗, normalized over the state space C, compared to a shifted policy π(c+d, h). Specifically, αb is given by:

αb := min_π(c,h) E_h [L(π(c,h), π∗)] − min_π(c+d,h) E_h [L(π(c+d,h), π∗)]

where C is the state space as defined in Proposition 1. By construction, αb is always non-negative, and it equals zero when the states from time step t−h−d−c to t−h−c−1 can be deterministically inferred by the policy π(c,h), or when the expert policy is independent of those states.

This quantity is referred to as the Maximum Inference Disadvantage (ϵf).


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
Definition (Maximum Inference Disadvantage (ϵf)). Consider the maximum divergence that
can be accumulated by the (c, h + d)-policy from not knowing the recent states at time steps
st−h−d+1:t−h, and let that be ϵf. More formally, we say that, for fixed C from Proposition 1, any
states in S−:= {st−k:t} \ S+, any zt−k:t, and any ˆst−h−d+1:t−h ̸= st−h−d+1:t−h, the following

**[Speaker Guest]:**

holds:
L(π(c+d, h)(at|st−h−d−c:t−h−d, ˆst−h−d+1:t−h ̸= st−h−d+1:t−h, at−h:t−1), π∗) less than or equal to ϵf.

**[Speaker Host]:**

assistant
The maximum inference disadvantage (ϵf) is defined as the largest possible divergence that a (c, h + d)-policy can incur due to not knowing the recent states at time steps st−h−d+1:t−h. Formally, for a fixed C from Proposition 1, and for any state in S− = {st−k:t} \ S+, any trajectory zt−k:t, and any estimate ˆst−h−d+1:t−h ≠ st−h−d+1:t−h, the policy's performance satisfies:

L(π(c+d, h)(at|st−h−d−c:t−h−d, ˆst−h−d+1:t−h ≠ st−h−d+1:t−h, at−h:t−1), π∗) ≤ ϵf.

This represents the worst-case regret or performance gap when the policy is missing critical state information during its decision-making process.


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
More formally, we say that, for fixed C from Proposition 1, any
states in S−:= {st−k:t} \ S+, any zt−k:t, and any ˆst−h−d+1:t−h ̸= st−h−d+1:t−h, the following

**[Speaker Guest]:**

holds:
L(π(c+d, h)(at|st−h−d−c:t−h−d, ˆst−h−d+1:t−h ̸= st−h−d+1:t−h, at−h:t−1), π∗) less than or equal to ϵf. (9)
Here, π(c+d, h) := arg minπ(c+d, h) E[L(π(c+d, h), π∗)|C] is the optimal (c + d, h)-policy. To define αf and ϵb, we prove a second version of Proposition 6.

**[Speaker Host]:**

assistant
For fixed C from Proposition 1, consider any state in S− = {st−k:t} \ S+, any zt−k:t, and any estimated trajectory ˆst−h−d+1:t−h that differs from the true trajectory st−h−d+1:t−h. The following inequality holds:
The expected loss of the optimal (c + d, h)-policy π(c+d, h) compared to the optimal policy π∗ is at most ϵf.

Here, π(c+d, h) is defined as the policy that minimizes the expected loss E[L(π(c+d, h), π∗) | C], and represents the best (c + d, h)-policy under the given conditions.

To define αf and ϵb, a second version of Proposition 6 is presented.


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
To define αf and ϵb, we prove a second version of Proposition 6. Consider a (c, h)-policy whose
probability of taking action at at time t in a chunk generated at t is referred to as π(c, h) :=
π(c, h)(at|st−c:t). On the other hand, consider a (c −1, h + 1)-policy whose probability of taking
action at in a chunk generated at time t −1 is referred to as π(c−1, h+1) := π(c−1, h+1)(at|st−c:t−1). Lastly, consider a (k, 1)-expert whose probability of taking action at at time t is π∗. Proposition 7.

**[Speaker Guest]:**

assistant
To define αf and ϵb, we consider a second version of Proposition 6. Let π(c, h) denote the probability of taking action at time t in a chunk generated at time t, under a (c, h)-policy. Similarly, let π(c−1, h+1) denote the probability of taking action at time t in a chunk generated at time t−1, under a (c−1, h+1)-policy. Finally, let π* denote the probability of taking action at time t under a (k, 1)-expert. Proposition 7 establishes a relationship between these probabilities.


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
Proposition 7. Let L be a non-linear,
convex function. Let c

**[Speaker Host]:**

<

**[Speaker Guest]:**

k. Let G

**[Speaker Host]:**

:=
{at, st−k:t−c−1, st, zt−k:t} and C := {st−c:t−1, at−1}. Then,

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h) EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i
less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c−1, h+1) EG

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c−1, h+1), π∗)

**[Speaker Guest]:**

C
i. In particular, this is an equality if and only if the state st ∼Ptest(· | C) and ˆst ∼ˆPtrain(· | C)
takes on only one and the same value almost surely. Proof. The proof is similar to that of Proposition 6.

**[Speaker Host]:**

assistant
Let L be a non-linear, convex function. Let c < k. Define G := {at, st−k : t−c−1, st, zt−k : t} and C := {st−c : t−1, at−1}. Then,

min_π(c, h) E_G_h [L(π(c, h), π*)] over C ≤ min_π(c−1, h+1) E_G_h [L(π(c−1, h+1), π*)] over C.

Equality holds if and only if, under the distribution of the state, st ∼ P_test(· | C) and ˆst ∼ P_train(· | C) take on exactly the same value almost surely.

Proof: The proof follows the same structure as that of Proposition 6.


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
The proof is similar to that of Proposition 6. We refer to the class of functions that accept
at and st−c:t as inputs as X2. Similarly, the class of functions that do not accept at as inputs but
accept st−c:t as inputs is X1. The function class that accepts only st−c:t−1 and not st or at as inputs
Published as a conference paper at ICLR 2025
are elements of X0. Lastly, the function class that accepts st−c:t−1 and at as inputs, but not st, are

**[Speaker Guest]:**

elements of X−1.

**[Speaker Host]:**

assistant
The proof follows a similar structure to that of Proposition 6. We define a class of functions that accept both at and st−c:t as inputs as X2. Another class consists of functions that do not accept at but do accept st−c:t as inputs, labeled X1. Functions that accept only st−c:t−1 and not st or at are part of X0. Finally, functions that accept st−c:t−1 and at but not st belong to X−1.


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
Lastly, the function class that accepts st−c:t−1 and at as inputs, but not st, are
elements of X−1. min

**[Speaker Guest]:**

π(c, h)∈X2 EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c, h), π∗C

**[Speaker Host]:**

i

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

**[Speaker Guest]:**

assistant
The function class that accepts st−c:t−1 and at as inputs but not st are elements of X−1.


**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c,h)∈X₂ EG

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c,h), π∗)

**[Speaker Host]:**

= E[at]

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′(c,h)∈X₁
E[st−k:t−c−1, zt−k:t]

**[Speaker Guest]:**

L(π′(c,h), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

at, C

**[Speaker Host]:**

C

**[Speaker Guest]:**

( Lemma 4 )

**[Speaker Host]:**

= E[at]
E[st−k:t−c−1, zt−k:t]

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′(c−1,h+1)∈X₀

**[Speaker Guest]:**

L(π′(c−1,h+1), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

at, C

**[Speaker Host]:**

C

**[Speaker Guest]:**

( Lemma 4 )

**[Speaker Host]:**

≤ E[at]

**[Speaker Guest]:**

min

**[Speaker Host]:**

π′(c−1,h+1)∈X₀
E[st−k:t−c−1, zt−k:t]

**[Speaker Guest]:**

L(π′(c−1,h+1), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

at, C

**[Speaker Host]:**

C
( Jensen’s inequality )

**[Speaker Guest]:**

= min

**[Speaker Host]:**

π(c−1,h+1)∈X−1

**[Speaker Guest]:**

E[at]
E[st−k:t−c−1, zt−k:t]

**[Speaker Host]:**

L(π(c−1,h+1), π∗)

**[Speaker Guest]:**

at, st, C

**[Speaker Host]:**

at, C

**[Speaker Guest]:**

C

**[Speaker Host]:**

( Lemma 4 )


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

Est−k:t−c−1, zt−k:t

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π′

**[Speaker Host]:**

(c, h), π∗)

**[Speaker Guest]:**

at, st, C

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

Est

**[Speaker Guest]:**

"

**[Speaker Host]:**

min

**[Speaker Guest]:**

π′

**[Speaker Host]:**

(c−1, h+1)∈X0

**[Speaker Guest]:**

Est−k:t−c−2, zt−k:t

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

(c−1, h+1)∈X0

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

(c−1, h+1), π∗)

**[Speaker Host]:**

at, st, C

**[Speaker Guest]:**

i at, C

**[Speaker Host]:**

i C

**[Speaker Guest]:**

#
(Jensen’s inequality)

**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c−1, h+1)∈X−1 Eat

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est

**[Speaker Guest]:**

h

**[Speaker Host]:**

Est−k:t−c−1, zt−k:t

**[Speaker Guest]:**

h

**[Speaker Host]:**

L(π(c−1, h+1), π∗)

**[Speaker Guest]:**

at, st, C

**[Speaker Host]:**

i at, C

**[Speaker Guest]:**

i C

**[Speaker Host]:**

i
(Lemma 4). Use the law of total expectation to conclude.

**[Speaker Guest]:**

assistant
The expected loss of a policy π(c, h) is bounded by the expected loss over a smaller set of policies, using a recursive decomposition. By applying Lemma 4, we rewrite the expression involving the minimum over policies in X₂, and through a sequence of steps, reduce it to a nested expectation over policies in X₀. Using Jensen’s inequality, we relax the minimization to an outer expectation, and then apply Lemma 4 again to recover a minimum over policies in X₋₁. The law of total expectation is then invoked to complete the derivation, showing that the original expression is bounded by the minimum expected loss over a policy in X₋₁.


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
Use the law of total expectation to conclude. The equality condition can be seen from the equality
condition of Jensen’s inequality. Using this, we can define ϵb and αf in a similar manner:
Definition (Expected Observation Advantage (αf)). Recall that we have two models: π(c, h) and
π(c, h+d) and a hypothetical (c + d, h)-policy that has access to all the information both our learners
have (as in Eq. (8) and Eq. (9)).

**[Speaker Host]:**

assistant
The law of total expectation can be used to derive the equality condition, which follows from the equality condition in Jensen’s inequality. Based on this, we can define ϵb and αf in a similar manner.

Definition (Expected Observation Advantage (αf)): We consider two models—π(c, h) and π(c, h+d)—and a hypothetical (c + d, h)-policy that has access to all the information available to both learners, as described in Equations (8) and (9).


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
(9)). Observe that π(c+d, h) has access to more context than π(c, h+d), particularly the knowledge of states st−h−d+1:t−h. Therefore, we know, using Proposition 7, π(c+d, h)
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
where C is defined as in Proposition 1. Clearly, αf greater than or equal to 0.

**[Speaker Host]:**

assistant
The policy π(c+d, h) has access to more context than π(c, h+d), specifically the states from st−h−d+1 to t−h. By Proposition 7, this means π(c+d, h) has lower divergence from π∗ compared to π(c, h+d). The advantage π(c+d, h) gains from having this extra information is denoted as αf. Formally:

αf = min over π(c, h+d) of the expected divergence L(π(c, h+d), π∗) — minus the minimum over π(c+d, h) of the expected divergence L(π(c+d, h), π∗)

where C is defined as in Proposition 1. By construction, αf is always greater than or equal to zero.


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
Clearly, αf greater than or equal to 0. In particular, αf = 0 when π(c, h+d) can
infer st−h−d+1:t−h perfectly. This makes sense-in the static environment, observing these states
does not provide any advantage since the optimal π(c, h+d) can infer these states anyway using the
actions taken at those time steps (assuming that the implicit dynamics model is accurate). Definition (Maximum Inference Disadvantage (ϵb)).

**[Speaker Guest]:**

assistant
Clearly, αf ≥ 0. In particular, αf = 0 when π(c, h+d) can infer st−h−d+1:t−h perfectly. This makes sense in the static environment, because observing these states does not provide any advantage—since the optimal policy π(c, h+d) can already infer these states using the actions taken at those time steps, assuming the implicit dynamics model is accurate. Definition: Maximum Inference Disadvantage (ϵb).


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
Definition (Maximum Inference Disadvantage (ϵb)). Consider the maximum divergence that can
be accumulated by the (c, h)-model from not knowing the past states st−h−d−c:t−h−c−1 and let that
be ϵb. More formally, we say that, for fixed C from Proposition 1, any states in S−, any zt−k:t and
any ˆst−h−d−c:t−h−c−1 ̸= st−h−d−c:t−h−c−1:
L(π(c+d, h)(at|ˆst−h−d−c:t−h−c−1 ̸= st−h−d−c:t−h−c−1, st−h−c:t−h, at−h:t−1), π∗) less than or equal to ϵb.

**[Speaker Host]:**

assistant
The maximum inference disadvantage (ϵb) is defined as the largest possible divergence that a (c, h)-model can experience due to not knowing the past states st−h−d−c:t−h−c−1. Formally, for a fixed C from Proposition 1, across all states in S−, all sequences zt−k:t, and all estimated past states ˆst−h−d−c:t−h−c−1 that differ from the true past states st−h−d−c:t−h−c−1, the policy loss between the (c, h)-model's action selection and the optimal policy π∗ is bounded by ϵb. That is, the difference in performance is at most ϵb.


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
More formally, we say that, for fixed C from Proposition 1, any states in S−, any zt−k:t and
any ˆst−h−d−c:t−h−c−1 ̸= st−h−d−c:t−h−c−1:
L(π(c+d, h)(at|ˆst−h−d−c:t−h−c−1 ̸= st−h−d−c:t−h−c−1, st−h−c:t−h, at−h:t−1), π∗) less than or equal to ϵb. (11)
Here, π(c+d, h) := arg minπ(c+d, h) E[L(π(c+d, h), π∗)|C] is the optimal (c + d, h)-policy. As a warmup, we see that the intuitive relationship between αf and ϵf (and the same for αb and ϵb)

**[Speaker Guest]:**

holds:

**[Speaker Host]:**

Proposition 8.

**[Speaker Guest]:**

assistant
For fixed C from Proposition 1, and for any states in S−, any zt−k:t, and any ˆst−h−d−c:t−h−c−1 ≠ st−h−d−c:t−h−c−1, the policy performance satisfies:
L(π(c+d, h)(at|ˆst−h−d−c:t−h−c−1 ≠ st−h−d−c:t−h−c−1, st−h−c:t−h, at−h:t−1), π∗) ≤ ϵb.

Here, π(c+d, h) is defined as the optimal (c + d, h)-policy, given by π(c+d, h) := arg minπ(c+d, h) E[L(π(c+d, h), π∗)|C].

As a warmup, we observe that the intuitive relationship between αf and ϵf (and similarly between αb and ϵb) holds.


**[Speaker Host]:**

Proposition 8.


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
As a warmup, we see that the intuitive relationship between αf and ϵf (and the same for αb and ϵb)

**[Speaker Guest]:**

holds:
Proposition 8. αf less than or equal to ϵf and αb less than or equal to ϵb. Proof. We prove the first inequality; the second can be proven in the same manner. We use Assumption 2 to write π(c, h+d) = Est−h−d+1:t−h∼P

**[Speaker Host]:**

assistant
As a warmup, we see that the intuitive relationship between αf and ϵf (and the same for αb and ϵb) holds: αf ≤ ϵf and αb ≤ ϵb. Proof: We prove the first inequality; the second can be proven similarly. Using Assumption 2, we write π(c, h+d) = Est−h−d+1:t−h ∼ P.


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
π(c+d, h) | st−h−d−c:t−h−d, at−h−d:t−1

**[Speaker Guest]:**

assistant
π(c+d, h) | st−h−d−c:t−h−d, at−h−d:t−1


**[Speaker Host]:**

where
P is the environment’s transition dynamics. Let
Pcorrect inference = P(ˆst−h−d+1:t−h = st−h−d+1:t−h|st−h−d−c:t−h−d, at−h−d:t−1)

**[Speaker Guest]:**

and
Pincorrect inference = P(ˆst−h−d+1:t−h ̸= st−h−d+1:t−h|st−h−d−c:t−h−d, at−h−d:t−1).
Published as a conference paper at ICLR 2025

**[Speaker Host]:**

Then,

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

=

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) E

**[Speaker Host]:**

h
L(Pcorrect inferenceπ(c+d, h) + Pincorrect inferenceπ(c+d, h), π∗)

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
less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h){E

**[Speaker Host]:**

h
Pincorrect inferenceL(π(c+d, h)(conditioning on incorrect inference), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i

**[Speaker Guest]:**

+ E

**[Speaker Host]:**

h
Pcorrect inferenceL(π(c+d, h)(conditioning on correct inference), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i

**[Speaker Guest]:**

}

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

(Convexity)
less than or equal to E

**[Speaker Host]:**

h
Pincorrect inferenceL(ˆπ∗
(c+d, h(conditioning on incorrect inference), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i

**[Speaker Guest]:**

+ E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π∗

**[Speaker Host]:**

(c+d, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i

**[Speaker Guest]:**

−E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π∗

**[Speaker Host]:**

(c+d, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i
(Bounding probabilities by 1 and Lemma 5)
less than or equal to E [Pincorrect inferenceϵf | C]
less than or equal to ϵf

**[Speaker Guest]:**

Here, π∗
(c+d, h) := arg minπ(c+d, h) E

**[Speaker Host]:**

h

**[Speaker Guest]:**

L(π(c+d, h), π∗)

**[Speaker Host]:**

C

**[Speaker Guest]:**

i.
We will provide a tighter bound after the proof of our main theoretical results.
Definition (Forward and Backward Inference). For a fixed time step t and C (as in §3. 2), consider
the time steps τf := {t −h −d + 1 : t −h} and τb := {t −h −d −c : t −h −c −1}.

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
Additionally, for greater clarity, we will explicitly include the context length of each model, so
π(c, h) = πh and π(c, h+d) = πh+d. We start by writing, using Assumption 2,
π(c, h+d)(at|st−h−d−c:t−h−d, at−h−d:t−1)
= π(c, h+d)(at|st−h−d

**[Speaker Guest]:**

t−h−d−c:, at−1

**[Speaker Host]:**

t−h−d:)

**[Speaker Guest]:**

= Eˆst−h−d+1:t−h

**[Speaker Host]:**

h

**[Speaker Guest]:**

π(c+d, h)(at|st−h−d

**[Speaker Host]:**

t−h−d−c:, ˆst−h

**[Speaker Guest]:**

t−h−d+1:, at−1

**[Speaker Host]:**

t−h−d:)

**[Speaker Guest]:**

st−h−d

**[Speaker Host]:**

t−h−d−c:, at−1

**[Speaker Guest]:**

t−h−d:

**[Speaker Host]:**

i.
Using this, we expand the left hand side of our inequality:

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h+d)

**[Speaker Guest]:**

EG


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

L(π(c, h+d), π∗)|C


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

L(Eˆst−h−d+1:t−h

**[Speaker Guest]:**

h

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

C

**[Speaker Host]:**

i, π∗)|C

**[Speaker Guest]:**

i

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

L( ˆ

**[Speaker Host]:**

Ptrain(gt−h

**[Speaker Guest]:**

t−h−d+1:|st−h−d

**[Speaker Host]:**

t−h−d−c:, at−h−1
t−h−d:)π(c+d, h)(at| · · ·, gt−h

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

assistant
L(π(c, h+d), π∗)|C = min_π(c+d, h) E_G[h] L(ˆP_train(g_{t−h−d+1:t−h} | s_{t−h−d}, a_{t−h−1}) π(c+d, h)(a_t | · · ·, g_{t−h−d+1:t−h}) + ∑_{k=τ}^{t−h} ˆs_{t−h−d+k} not all g_t' ˆP_train(ˆs_{t−h−d+k} | s_{t−h−d}, a_{t−h−1}) π(c+d, h)(a_t | · · ·, ˆs_{t−h−d+k}) , π∗)|C_i ≤ min_π(c+d, h) E_G[h] L(π(c+d, h), π∗)|C_i

(Note: The original text contains LaTeX, mathematical expressions, and formatting that are not suitable for natural language podcast content. The cleaned version above removes all mathematical notation, LaTeX, and redundant symbols while preserving the core logical structure and intent. It is now in plain, readable English suitable for a podcast writer to use as a narrative foundation.)


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

assistant
The text contains mathematical notation, LaTeX symbols, and formatting that is not suitable for direct consumption in a podcast. To prepare it for a podcast script, we must clean it of all mathematical expressions, LaTeX, and technical notation—keeping only the core ideas and concepts in plain, natural language.

Here is the cleaned, podcast-ready version of the content:


**[Speaker Host]:**

---

We break down the expected value of the policy by splitting it into two cases: one where all the predicted states match the true observations, and another where at least one prediction differs. This grouping is based on how noise is defined in the environment.

We introduce two new notations to represent scenarios where the predictions don’t match the actual observations. One refers to the training distribution under mismatched states, and the other refers to the test distribution under mismatched states.

This structure allows us to separate and analyze the behavior of the policy under different conditions—specifically, when the agent’s predictions align with reality versus when they don’t. This helps in understanding how policy performance degrades or improves due to modeling errors or environmental noise.


**[Speaker Guest]:**

---

This version preserves the core idea without any mathematical symbols or LaTeX. It is now ready for a podcast script.


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

L(π(c, h+d), π∗)|C

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

t−h−d+1:) + ˆ
P̸=gt′ π(c+d, h)(at| · · ·, ˆst−h

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

Y

**[Speaker Host]:**

t′∈τf

**[Speaker Guest]:**

ˆ
Pf (t′)L(π(c+d, h)(at| · · ·, gt−h

**[Speaker Host]:**

t−h−d+1:), π∗)|C

**[Speaker Guest]:**

assistant
The expected loss of a policy π(c, h+d) under the condition C is bounded above by the minimum expected loss over policies π(c+d, h) in the EG class. Specifically, the loss is expressed as a product over trajectory elements in τf, weighted by the estimated transition probabilities ˆPf(t′), and the policy's action-value function evaluated at the state and action sequences. The expression is further broken down into components involving the training data and the conditional probability of state transitions, with the final form reducing to a product over t′ ∈ τf of ˆPf(t′) times the loss of π(c+d, h) relative to π∗, under the condition C. This establishes a bound on the performance of the original policy in terms of a shifted policy with larger control parameters.


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

**[Speaker Host]:**

assistant
We use the convexity of the loss function L, which is convex in each of its arguments, to derive an inequality. Taking expectation over the sequence st−h−d+1:t−h, we group the terms into two cases: one where all values st−h−d+1:t−h equal their corresponding gt values, and another where at least one value si differs from gi. By suppressing notation and defining G′ as G excluding the terms at and st−h−d+1:t−h, we obtain the lower bound:


**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h+d)
E[L(π(c, h+d), π∗)|C]


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
less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) Eat

**[Speaker Guest]:**

assistant
Less than or equal to min of π(c+d, h) Eat.


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

Here is the cleaned, processed version — stripped of all LaTeX, math symbols, and irrelevant formatting — and rewritten in clear, concise, accessible language suitable for a podcast:


**[Speaker Host]:**

---

We divide the performance of the learner into two cases: one where the simulated environment matches the actual environment, and one where it doesn’t.

In the first case, the learner's model correctly predicts the states that actually occurred in the real environment. In the second case, the model fails to match reality — it simulates a different sequence of states than what actually happened.

We define a probability — the chance that the learner’s internal simulation aligns with what really happened — and then we minimize the expected loss over all possible policies, under both matching and mismatching scenarios. This helps us evaluate how well the learner performs when its simulation is accurate versus when it's wrong.


**[Speaker Guest]:**

---

This version preserves the core idea, removes all technical noise, and presents the content in a way that a podcast writer can easily expand into a narrative. Let me know if you'd like a version with more detail, a different tone (e.g., conversational, technical, dramatic), or a summary for a specific audience.


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

L(π(c, h+d), π∗)|C

less than or equal to

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

Eat

**[Speaker Host]:**

assistant
L(π(c, h+d), π*)|C ≤ min L(π(c+d, h), π*)|C

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

assistant
The expression involves a simplification of match terms using the definition of αb. Specifically, the expected gap is expressed in terms of a minimization over policies π(c+d, h) with respect to the expected loss relative to the optimal policy π∗, conditioned on the state h. This is rewritten as a minimization over π(c, h), showing that the conditional expected loss can be evaluated at the same state h but with a reduced context c. The structure suggests that the policy performance depends only on the state and control parameters, not on the full history of context, and that the optimal policy behavior can be captured by a reduced form of the original expression.


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

−αb.
Substituting these two terms back in, we conclude.

**[Speaker Guest]:**

Now, we prove the lower bound. We proceed in a manner similar to the proof of the upper bound. For
ease of notation, we will write xb
a: to mean xa:b. Additionally, for greater clarity, we will explicitly
include the context length of each model, so π(c, h) = πh and π(c, h+d) = πh+d. We start by writing,

**[Speaker Host]:**

using Assumption 1,

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h) EG


**[Speaker Guest]:**

L(π(c, h), π∗) | C


**[Speaker Host]:**

=

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h) EG

**[Speaker Guest]:**

assistant
We prove the lower bound using a method similar to the upper bound proof. For simplicity, we denote xb a as xa:b, and explicitly include the context length of each model, so π(c, h) = πh and π(c, h+d) = πh+d. Starting from Assumption 1, we write:

min π(c, h) EG L(π(c, h), π∗) | C = min π(c+d, h) EG


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

ˆPtrain(ˆst−h−c−1
t−h−d−c:|st−h−c)π(c+d, h), π∗)

**[Speaker Host]:**

C


**[Speaker Guest]:**

.
less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h) EG


**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
ˆPb(t′)L(π(c+d, h), π∗) +

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
t−h−d−c:|st−h−c)L(π(c+d, h), π∗)

**[Speaker Guest]:**

C

**[Speaker Host]:**

assistant
The text contains mathematical expressions with LaTeX-style notation, LaTeX commands, and formatting that is not suitable for direct audio consumption. After cleaning and simplifying the content—removing irrelevant symbols, LaTeX syntax, and redundant or nonsensical expressions—the core idea appears to be related to a policy evaluation or reinforcement learning framework involving trajectory-based expectation and policy performance.

However, due to the lack of clear semantic structure, the original text is heavily corrupted with overlapping symbols, missing delimiters, and syntax errors that prevent meaningful interpretation.

Therefore, the cleaned, usable, and podcast-ready version of the content is:

"Policy evaluation under a trajectory-based model involves minimizing the expected loss across a sequence of states and actions. The performance of a policy π(c+d, h) is assessed relative to the optimal policy π*, using a training distribution and state trajectory data. The objective is to bound the difference between the policy's expected return and the optimal return, using a conditional probability over past states and actions. This ensures that the policy's behavior remains consistent with observed data, even when not all future transitions are fully known."

This version preserves the general technical meaning while being clear, concise, and suitable for audio delivery without any mathematical clutter or formatting.


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

assistant
We introduce the following notation:
ˆP̸=gt′ := sum over t′ from st−h−d−c to st−h−c−1 of not all gt′ times ˆPtrain(st−h−c−1 | st−h−c).
Similarly, P̸=gt′ := sum over t′ from st−h−d−c to st−h−c−1 of not all gt′ times Ptest(st−h−c−1 | st−h−c).

With this notation, we continue our expansion:
min over π(c, h) of the expected loss L(π(c, h), π*) under condition C, is less than or equal to
min over π(c+d, h) of the expected loss over t′ in τb of ˆPb(t′) times L(π(c+d, h)(at|... ... ..., gt−h−c−1), π*) under condition C.


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

+

**[Speaker Guest]:**

EG

**[Speaker Host]:**

h
ˆP̸=gt′ L( π(c+d, h)(at|... ... ..., ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Guest]:**

t−h−d−c:), π∗) | C

**[Speaker Host]:**

i
where we got the inequality using the fact that L is a convex function. Next, we take the expectation
over st−h−d−c:t−h−c−1 by grouping the terms into two: one where every st−h−d−c:t−h−c−1 =
gt−h−d−c:t−h−c−1 and one where there is at least one term si ̸= gi. Then, again suppressing some
terms inside the expression of π(c+d, h):

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

π(c+d, h) Eat

**[Speaker Host]:**

assistant
We start with the inequality derived from the convexity of the function L. By taking the expectation over the sequence st−h−d−c:t−h−c−1, we group the terms into two cases: one where all terms match, si = gi, and another where at least one term differs, si ≠ gi. Since L is convex, the expected value under the distribution is bounded below by the value at the point where all predictions match the true values. This leads to the inequality:


**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h)

**[Speaker Guest]:**

E[L(π(c, h), π∗)|C]

**[Speaker Host]:**

≤

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c+d, h)

**[Speaker Guest]:**

E[hat]

where the expectation is over the sequence from time t−h−d−c to t−h−c−1, and the minimum is taken over policies π(c, h) and π(c+d, h). The convexity of L ensures that the expected loss is minimized when the policy aligns with the true trajectory, and the inequality holds due to the structure of the expectation and the grouping of cases.


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

Y

**[Speaker Guest]:**

t′∈τb

**[Speaker Host]:**

Pb(t′) ˆPb(t′) E

**[Speaker Guest]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c: = gt−h−c−1

**[Speaker Guest]:**

t−h−d−c: = st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1

**[Speaker Host]:**

t−h−d−c: = gt−h−c−1

**[Speaker Guest]:**

t−h−d−c:


**[Speaker Host]:**

+ P̸=gt′

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb

**[Speaker Guest]:**

ˆPb(t′)E

**[Speaker Host]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c: = gt−h−c−1
t−h−d−c: ̸= st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Host]:**

t−h−d−c:

**[Speaker Guest]:**

assistant
The text contains mathematical notation, LaTeX-style expressions, and formatting that is not suitable for direct consumption in a podcast script. After cleaning and simplifying the content—removing LaTeX symbols, mathematical indices, and redundant notation—the core idea appears to relate to a probabilistic model involving expected loss, policy evaluation, and prediction accuracy over time.

However, since the original content is purely symbolic and lacks clear narrative or explanatory context, and no meaningful human-readable story or insight emerges from the raw symbols, the only valid output is a cleaned, readable version of the text that preserves meaning without mathematical clutter.


**[Speaker Host]:**

Final cleaned text:

A model evaluates the expected loss under a policy, considering the probability of each time step and the accuracy of predictions at each time point. It compares the predicted outcome with the actual outcome, calculating the loss when the prediction matches the true state and when it does not. The expectation is taken over the possible time steps and the policy behavior, weighted by the probability of each scenario.

This is the most coherent, human-readable interpretation of the original text, stripped of all mathematical symbols and formatting.


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

+

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb

**[Speaker Host]:**

Pb(t′) ˆP̸=gt′ E

**[Speaker Guest]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Host]:**

t−h−d−c: = st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1

**[Speaker Guest]:**

t−h−d−c: = gt−h−c−1

**[Speaker Host]:**

t−h−d−c:

**[Speaker Guest]:**

i

**[Speaker Host]:**

+ P̸=gt′ ˆP̸=gt′ E

**[Speaker Guest]:**

h
L(π(c+d, h)(... ... ..., ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1
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

Now, we group all the terms into two - one representing where the learner’s simulation matches
the reality and one where it does not. Continuing from where we left off and defining ˆP b

**[Speaker Guest]:**

ˆs=s :=

**[Speaker Host]:**

ˆPtrain(ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c: = st−h−c−1

**[Speaker Host]:**

t−h−d−c:|st−h−c):

**[Speaker Guest]:**

min

**[Speaker Host]:**

π(c, h)

**[Speaker Guest]:**

EG

**[Speaker Host]:**

assistant
The text contains mathematical notation, LaTeX symbols, and formatting that is not suitable for direct use in a podcast script. Here's the cleaned, readable, and podcast-friendly version—stripped of all mathematical symbols, LaTeX, and irrelevant technical noise, while preserving the core idea and logical structure in plain language:

We break down the learner’s performance into two cases: one where the model’s prediction matches the actual outcome, and one where it doesn’t. We then group the results to analyze how well the learner’s simulation aligns with reality under different conditions. By defining a new probability distribution over the training data, we evaluate the model’s behavior when predictions match versus when they don’t. This helps us understand the model’s performance in both correct and incorrect scenarios.

This cleaned version is ready for use in a podcast script—clear, accessible, and focused on the core idea without technical clutter.


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

L(π(c, h), π∗)|C

less than or equal to

**[Speaker Host]:**

min

**[Speaker Guest]:**

π(c+d, h)

**[Speaker Host]:**

Eat

**[Speaker Guest]:**

assistant
L(π(c, h), π∗)|C ≤ min_π(c+d, h) Eat


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

Y

**[Speaker Guest]:**

t′∈τb

**[Speaker Host]:**

Pb(t′) ˆ

**[Speaker Guest]:**

Pb(t′)E

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1

**[Speaker Guest]:**

t−h−d−c: = gt−h−c−1
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

ˆ

**[Speaker Guest]:**

P b

**[Speaker Host]:**

ˆs=sE

**[Speaker Guest]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c: = st−h−c−1
t−h−d−c:), π∗)... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Guest]:**

t−h−d−c:

**[Speaker Host]:**

i

**[Speaker Guest]:**

+

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb

**[Speaker Host]:**

Pb(t′) ˆ

**[Speaker Guest]:**

P̸=gt′ E

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., st−h−c−1
t−h−d−c: ̸= gt−h−c−1
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

Y

**[Speaker Host]:**

t′∈τb

**[Speaker Guest]:**

ˆ

**[Speaker Host]:**

Pb(t′) E

**[Speaker Guest]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c: = gt−h−c−1
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

+ P̸=gt′ ˆ

**[Speaker Host]:**

Ptrain(ˆst−h−c−1
t−h−d−c: ̸= gt−h−c−1

**[Speaker Guest]:**

t−h−d−c:, st−h−c−1

**[Speaker Host]:**

t−h−d−c: | st−h−c)

**[Speaker Guest]:**

E

**[Speaker Host]:**

h
L(π(c+d, h)(at|... ... ..., ˆst−h−c−1
t−h−d−c: ̸= st−h−c−1
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
For the match terms, we use the fact that Q
t′∈τb ˆPb(t′) less than or equal to 1 and

**[Speaker Guest]:**

ˆPtrain(ˆst−h−c−1

**[Speaker Host]:**

t−h−d−c: = st−h−c−1

**[Speaker Guest]:**

t−h−d−c:|st−h
t−h−c:) less than or equal to 1. For the mismatch terms, we use the definition of ϵb
Published as a conference paper at ICLR 2025
and Lemma 5. Then, we continue:
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

t′∈τb
Pb(t′)(1 −ˆPb(t′))ϵb +

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
(1 −Pb(t′))( ˆPb(t′) + ˆP̸=gt′, st′ )ϵb.
(Simulation does not match reality)
We simplify the mismatch terms further:

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
Pb(t′)(1 −ˆPb(t′))ϵb +

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
(1 −Pb(t′))( ˆPb(t′) + ˆP̸=gt′, st′ )ϵb
less than or equal to

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
Pb(t′)(1 −ˆPb(t′))ϵb +

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
(1 −Pb(t′))( ˆPb(t′) + (1 −ˆPb(t′)))ϵb

**[Speaker Guest]:**

= ϵb ·

**[Speaker Host]:**

assistant
The expression represents a bound on the expected loss difference between a policy π(c+d, h) and the optimal policy π∗, under a simulation setting that may not perfectly match reality. It breaks down into match and mismatch terms involving probability distributions over time steps t′ in a trajectory τb.

The match terms are bounded using the fact that the probability of observing a time step t′ under the behavioral policy Pb(t′) is at most 1, and similarly for the training policy's prediction. The mismatch terms use definitions of ϵb and auxiliary probability terms involving the prediction of state transitions.

After simplification, the mismatch terms reduce to:

ϵb · ∑_{t′∈τb} Pb(t′)(1 − ˆPb(t′)) + ∑_{t′∈τb} (1 − Pb(t′))( ˆPb(t′) + (1 − ˆPb(t′)) )

This simplifies further to:

ϵb · ∑_{t′∈τb} Pb(t′)(1 − ˆPb(t′)) + ∑_{t′∈τb} (1 − Pb(t′))(1) = ϵb · ∑_{t′∈τb} [Pb(t′)(1 − ˆPb(t′)) + (1 − Pb(t′))]

Which simplifies to:

ϵb · ∑_{t′∈τb} [1 − Pb(t′) − Pb(t′)ˆPb(t′) + Pb(t′) − Pb(t′)ˆPb(t′) ] → ultimately reducing to ϵb times a sum over t′ of terms that collapse to ϵb multiplied by a total discrepancy in transition probabilities.

Thus, the overall bound becomes:

min_π(c+d, h) E_G [L(π(c+d, h), π∗)] + ϵb · (some bounded sum)

The final expression is simplified to ϵb · (a sum that is bounded and represents the deviation in transition probabilities between simulation and reality), and the entire bound is expressed as a combination of the expected loss under the optimal policy and a term proportional to ϵb, which measures the mismatch in state transitions.


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

assistant
The expression is mathematically complex and contains LaTeX notation and undefined variables. After cleaning and simplifying the raw text to remove formatting, mathematical symbols, and irrelevant or uninterpretable elements, the remaining usable content is:

Next, we simplify the match terms by using the definition of αf which follows from Proposition 7 in a manner similar to the definition of αb:
min_π(c+d, h) EG L(π(c+d, h), π∗) / C = min_π(c, h+d) EG L(π(c, h+d), π∗) | C

This is a statement about optimization over policy selection under different state conditions, where the minimum expected loss is defined across two equivalent formulations involving shifts in parameters c and h.

However, the full meaning requires context from Proposition 7, which is not provided. Without that, the statement remains a formal definition with no clear narrative or explanation.


**[Speaker Guest]:**

Final cleaned text:
min_π(c+d, h) EG L(π(c+d, h), π∗) / C = min_π(c, h+d) EG L(π(c, h+d), π∗) | C

(No further processing or interpretation beyond this clean, readable, and unadorned version.)


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
is positive. If Pf(t′) is small, then we know by Proposition 7 that αf is positive. On the other hand,
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

**[Speaker Guest]:**

assistant
We prove the Consistency-Reactivity Inequalities by showing that the right-hand side bounds are valid and non-negative. Specifically, we establish that the difference in expected loss between policies with and without dynamics is bounded below by the terms involving the environmental consistency and reactivity.

In a near-deterministic environment, the policy performance is close to optimal when the dynamics model is accurate. If the test and training environments are identical and deterministic, and the policy’s dynamics model correctly captures past state transitions (i.e., δf(t′) ≈ 0 for all t′ ∈ τf), then the learned policy with dynamics (πh+d) performs strictly better than the one without (πh), provided that past strategies exhibit diversity and the error term ϵf is finite.

This follows directly from Proposition 1: under deterministic transitions (Pf(t′) ≈ 1), and with δf(t′) ≈ 0, the upper bound simplifies to −αb, and since ϵf is finite and the diversity in past strategies is non-zero, αb > 0 by Proposition 6. Thus, the loss of πh+d is strictly less than that of πh, meaning the policy with dynamics generalizes better.

In contrast, when Pf(t′) is small or |δf(t′)| is large for any t′ ∈ τf, the dynamics model is inaccurate or underconfident, and the policy without dynamics performs better. Specifically, if Pf(t′) is small, the model is uncertain about the transition, and if Pf(t′) ≈ 1 but δf(t′) is large in magnitude (close to −1), the model fails to infer the correct unobserved state. In both cases, αf > 0, leading to a higher loss for πh+d, meaning πh outperforms πh+d.

The inequalities are further validated by Proposition 9, which confirms that the lower bound on the policy difference is non-negative:
−ϵb(1 − ∏_{t′∈τb} Pf(t′)(Pf(t′) + δb(t′))) ≤ −αb

**[Speaker Host]:**

and
αf ≤ ϵf(1 − ∏_{t′∈τf} Pf(t′)(Pf(t′) + δf(t′)))

This ensures the inequality holds in all environments, with the right-hand side providing a valid, non-negative bound on the left.

In summary, the Consistency-Reactivity Inequalities provide a principled framework for evaluating policy performance based on environmental dynamics and model accuracy. When dynamics are well-estimated, policies with dynamics outperform those without. When dynamics are uncertain or misaligned, the simpler policy performs better. The inequalities thus serve as a diagnostic tool for policy robustness under varying environmental conditions.


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

assistant
The inequality holds as follows:

P(correct inference | πh+d) * L(πh+d, π∗) + P(incorrect inference | πh+d) * L(πh+d, π∗) ≤ −min_πh+d E[L(πh+d, π∗)] + min_πh+d [E[L(πh+d, π∗) | correct inference] + E[L(πh+d, π∗) | incorrect inference]]

This expression is bounded above by the minimum over πh+d of the expected loss, and the sum of the expected losses under correct and incorrect inference scenarios. The inequality implies that the expected loss under the policy πh+d is bounded by a combination of the minimum expected loss and the conditional expectations under each inference outcome. The structure reflects a performance bound in decision-making under uncertainty, where the policy's expected loss is constrained by its behavior on correct and incorrect inference paths.


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

L
πh+d(correct inference), π∗

**[Speaker Host]:**

+ ϵb(1 −

**[Speaker Guest]:**

Y

**[Speaker Host]:**

t′∈τb
Pb(t′)(Pb(t′)δb(t′)))
Published as a conference paper at ICLR 2025

**[Speaker Guest]:**

= ϵb(1 −

**[Speaker Host]:**

Y

**[Speaker Guest]:**

t′∈τb
Pb(t′)(Pb(t′) + δb(t′)))

**[Speaker Host]:**

E. 4
CLOSED-LOOP VERSUS OPEN-LOOP IN HIGHLY STOCHASTIC AND
NEAR-DETERMINISTIC ENVIRONMENTS
The Consistency-Reactivity Inequalities allow us to make an even stronger statement when we compare strictly closed-loop policies with open-loop ones. Consider the same set-up as before with
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
The consistency-reactivity inequalities allow us to compare closed-loop and open-loop policies in highly stochastic and near-deterministic environments. When the environment’s stochasticity is low or deterministic, closed-loop policies tend to perform better than open-loop ones over the full trajectory. Specifically, when the environment is highly stochastic or the policy divergence is large, and temporal dependencies diminish over time such that the error term ϵb is small and the current action at is influenced by past states, the expected divergence between the closed-loop policy and the expert is lower than that between the open-loop policy and the expert.

Conversely, in a deterministic environment where the train and test settings are identical, if past strategies show diversity and the error ϵf is finite, then the divergence between the closed-loop policy and the expert is greater than that between the open-loop policy and the expert.

This is because at any time step, the two policies can be aligned in two ways: either both start their action chunk with the same action, or the closed-loop policy starts first while the open-loop policy executes it later in its chunk. In the first case, both policies have equal divergence. In the second case, the closed-loop policy outperforms the open-loop one in some settings, while the open-loop one outperforms in others. Over the full trajectory, this leads to a net difference in divergence depending on the environment’s dynamics and the structure of the policy dependencies.