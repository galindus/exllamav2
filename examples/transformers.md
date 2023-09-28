Generative pre-trained transformers (GPT) are a type of large language model (LLM)[1][2][3] and a prominent framework for generative artificial intelligence.[4][5] The first GPT was introduced in 2018 by OpenAI.[6] GPT models are artificial neural networks that are based on the transformer architecture, pre-trained on large data sets of unlabelled text, and able to generate novel human-like content.[2][3] As of 2023, most LLMs have these characteristics[7] and are sometimes referred to broadly as GPTs.[8]

OpenAI has released very influential GPT foundation models that have been sequentially numbered, to comprise its "GPT-n" series.[9] Each of these was significantly more capable than the previous, due to increased size (number of trainable parameters) and training. The most recent of these, GPT-4, was released in March 2023. Such models have been the basis for their more task-specific GPT systems, including models fine-tuned for instruction following—which in turn power the ChatGPT chatbot service.[1]

The term "GPT" is also used in the names and descriptions of such models developed by others. For example, other GPT foundation models include a series of models created by EleutherAI,[10] and recently seven models created by Cerebras.[11] Also, companies in different industries have developed task-specific GPTs in their respective fields, such as Salesforce's "EinsteinGPT" (for CRM)[12] and Bloomberg's "BloombergGPT" (for finance).[13]
History[edit]
Initial developments[edit]

Generative pretraining (GP) was a long-established concept in machine learning applications,[14][15] but the transformer architecture was not available until 2017 when it was invented by employees at Google.[16] That development led to the emergence of large language models such as BERT in 2018[17] which was a pre-trained transformer (PT) but not designed to be generative (BERT was an "encoder-only" model).[18] Also around that time, in 2018, OpenAI published its article entitled "Improving Language Understanding by Generative Pre-Training," in which it introduced the first generative pre-trained transformer (GPT) system ("GPT-1").[19]

Prior to transformer-based architectures, the best-performing neural NLP (natural language processing) models commonly employed supervised learning from large amounts of manually-labeled data. The reliance on supervised learning limited their use on datasets that were not well-annotated, and also made it prohibitively expensive and time-consuming to train extremely large language models.[19]

The semi-supervised approach OpenAI employed to make a large-scale generative system—and was first to do with a transformer model—involved two stages: an unsupervised generative "pretraining" stage to set initial parameters using a language modeling objective, and a supervised discriminative "fine-tuning" stage to adapt these parameters to a target task.[19]
Later developments[edit]

Regarding more recent GPT foundation models, OpenAI published its first versions of GPT-3 in July 2020. There were three models, with 1B, 6.7B, 175B parameters, respectively named babbage, curie, and davinci (giving initials B, C, and D).

In July 2021, OpenAI published Codex, a task-specific GPT model targeted for programming applications. This was developed by fine-tuning a 12B parameter version of GPT-3 (different from previous GPT-3 models) using code from GitHub.[20]

In March 2022, OpenAI published two versions of GPT-3 that were fine-tuned for instruction-following (instruction-tuned), named davinci-instruct-beta (175B) and text-davinci-001.[21], and then started beta testing code-davinci-002.[22] text-davinci-002 was instruction-tuned from code-davinci-002. Both text-davinci-003 and ChatGPT were released in November 2022, with both building upon text-davinci-002 via reinforcement learning from human feedback (RLHF). text-davinci-003 is trained for following instructions (like its predecessors), whereas ChatGPT is further trained for conversational interaction with a human user.[23][24]

OpenAI's most recent GPT foundation model, GPT-4, was released on March 14, 2023. It can be accessed directly by users via a premium version of ChatGPT, and is available to developers for incorporation into other products and services via OpenAI's API. Other producers of GPT foundation models include EleutherAI (with a series of models starting in March 2021)[10] and Cerebras (with seven models released in March 2023).[11]
Foundational models[edit]

A foundational model is an AI model trained on broad data at scale such that it can be adapted to a wide range of downstream tasks.[25]

Thus far, the most notable GPT foundation models have been from OpenAI's GPT-n series. The most recent from that is GPT-4, for which OpenAI declined to publish the size or training details (citing "the competitive landscape and the safety implications of large-scale models").[26]
OpenAI's "GPT-n" series Model 	Architecture 	Parameter count 	Training data 	Release date 	Training cost
GPT-1 	12-level, 12-headed Transformer decoder (no encoder), followed by linear-softmax. 	117 million 	BookCorpus:[27] 4.5 GB of text, from 7000 unpublished books of various genres. 	June 11, 2018[6] 	"1 month on 8 GPUs",[6] or 1.7e19 FLOP.[28]
GPT-2 	GPT-1, but with modified normalization 	1.5 billion 	WebText: 40 GB of text, 8 million documents, from 45 million webpages upvoted on Reddit. 	February 14, 2019 (initial/limited version) and November 5, 2019 (full version)[29] 	"tens of petaflop/s-day",[30] or 1.5e21 FLOP.[28]
GPT-3 	GPT-2, but with modification to allow larger scaling 	175 billion[31] 	499 Billion tokens consisting of CommonCrawl (570 GB), WebText, English Wikipedia, and two books corpora (Books1 and Books2). 	May 28, 2020[30] 	3640 petaflop/s-day (Table D.1 [30]), or 3.1e23 FLOP.[28]
GPT-3.5 	Undisclosed 	175 billion[31] 	Undisclosed 	March 15, 2022 	Undisclosed
GPT-4 	Also trained with both text prediction and RLHF; accepts both text and images as input. Further details are not public.[26] 	Undisclosed 	Undisclosed 	March 14, 2023 	Undisclosed. Estimated 2.1e25 FLOP.[28]

Other such models include Google's PaLM, a broad foundation model that has been compared to GPT-3 and has recently been made available to developers via an API,[32][33] and Together's GPT-JT, which has been reported as the closest-performing open-source alternative to GPT-3 (and is derived from earlier open-source GPTs).[34] Meta AI (formerly Facebook) also has a generative transformer-based foundational large language model, known as LLaMA.[35]

Foundational GPTs can also employ modalities other than text, for input and/or output. GPT-4 is a multi-modal LLM that is capable of processing text and image input (though its output is limited to text).[36] Regarding multimodal output, some generative transformer-based models are used for text-to-image technologies such as diffusion[37] and parallel decoding.[38] Such kinds of models can serve as visual foundation models (VFMs) for developing downstream systems that can work with images.[39]
Task-specific models[edit]

A foundational GPT model can be further adapted to produce more targeted systems directed to specific tasks and/or subject-matter domains. Methods for such adaptation can include additional fine-tuning (beyond that done for the foundation model) as well as certain forms of prompt engineering.[40]

An important example of this is fine-tuning models to follow instructions, which is of course a fairly broad task but more targeted than a foundation model. In January 2022, OpenAI introduced "InstructGPT"—a series of models which were fine-tuned to follow instructions using a combination of supervised training and reinforcement learning from human feedback (RLHF) on base GPT-3 language models.[41][42] Advantages this had over the bare foundational models included higher accuracy, less negative/toxic sentiment, and generally better alignment with user needs. Hence, OpenAI began using this as the basis for its API service offerings.[43] Other instruction-tuned models have been released by others, including a fully open version.[44][45]

Another (related) kind of task-specific models are chatbots, which engage in human-like conversation. In November 2022, OpenAI launched ChatGPT—an online chat interface powered by an instruction-tuned language model trained in a similar fashion to InstructGPT.[46] They trained this model using RLHF, with human AI trainers providing conversations in which they played both the user and the AI, and mixed this new dialogue dataset with the InstructGPT dataset for a conversational format suitable for a chatbot. Other major chatbots currently include Microsoft's Bing Chat, which uses OpenAI's GPT-4 (as part of a broader close collaboration between OpenAI and Microsoft),[47] and Google's competing chatbot Bard (initially based on their LaMDA family of conversation-trained language models, with plans to switch to PaLM).[48]

Yet another kind of task that a GPT can be used for is the meta-task of generating its own instructions, like developing a series of prompts for 'itself' to be able to effectuate a more general goal given by a human user.[49] This is known as an AI agent, and more specifically a recursive one because it uses results from its previous self-instructions to help it form its subsequent prompts; the first major example of this was Auto-GPT (which uses OpenAI's GPT models), and others have since been developed as well.[50]

Multimodality[edit]

Generative transformer-based systems can also be targeted to tasks involving modalities beyond text.

For example, Microsoft’s “Visual ChatGPT” combines ChatGPT with visual foundation models (VFMs) to enable input or output comprising images as well as text.[51] Also, advances in text-to-speech technology offer powerful tools for audio content creation when used in conjunction with foundational GPT language models.[52]
Domain-specificity[edit]

GPT systems can be directed toward particular fields or domains. Some reported examples of such models and apps are as follows:

    EinsteinGPT - for sales and marketing domains, to aid with customer relationship management (uses GPT-3.5)[53]
    BloombergGPT - for the financial domain, to aid with financial news and information (uses "freely available" AI methods, combined with their proprietary data)[54]
    Khanmigo – described as a GPT version for tutoring, in the education domain, it aids students using Khan Academy by guiding them through their studies without directly providing answers (powered by GPT-4)[55][56]
    SlackGPT - for the Slack instant-messaging service, to aid with navigating and summarizing discussions on it (uses OpenAI's API)[57]
    BioGPT - for the biomedical domain, to aid with biomedical literature text generation and mining (uses GPT-2)[58]

Sometimes domain-specificity is accomplished via software plug-ins or add-ons. For example, several different companies have developed particular plugins that interact directly with OpenAI's ChatGPT interface,[59][60] and Google Workspace has available add-ons such as “GPT for Sheets and Docs”—which is reported to aid use of spreadsheet functionality in Google Sheets.[61][62]
Brand issues[edit]

OpenAI, which created the first generative pre-trained transformer (GPT) in 2018, has recently asserted that “GPT” should be regarded as a brand of OpenAI.[63] In April 2023, OpenAI revised the brand guidelines in its terms of service to indicate that other businesses using its API to run their artificial intelligence (AI) services would no longer be able to include “GPT” in such names or branding.[64] In May 2023, OpenAI engaged a brand management service to notify its API customers of this policy, although these notifications stopped short of making overt legal claims (such as allegations of trademark infringement or demands to cease and desist).[63]

Relatedly, OpenAI has applied to the United States Patent and Trademark Office (USPTO) to seek domestic trademark registration for the term “GPT” in the field of AI.[63] OpenAI sought to expedite handling of its application, but the USPTO declined that request in April 2023.[65] To get the trademark approved, OpenAI would need to establish that the term is actually “distinctive” to their specific offerings rather than widely understood as a broader technical term for the kind of technology. Some media reports suggested that OpenAI may be able to do so based indirectly on the fame of its GPT-based chatbot product, ChatGPT,[65][66] for which OpenAI has separately sought trademark protection (and which it has sought to enforce more strongly).[67] Other reports indicated that exclusivity for the bare term “GPT” seems unlikely to be granted,[63][68] as it is used frequently to refer simply to AI systems that involve generative pre-trained transformers.[3][69][70] If exclusive rights in the term were to be granted for the U.S., then everyone else in the U.S. using it in the name or branding of their related offerings would need to stop unless they have permission.[68] Even if that were to occur, the trademark doctrine of descriptive fair use could still preserve some room to continue non-brand-related usage.[71]
Selected bibliography[edit]
