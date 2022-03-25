# Awesome Egocentric

- [Surveys](#surveys) 
- [Papers](#papers)
    - [Episodic Memory](#episodic-memory)
        - [Moments Queries](#moments-queries)
    - [Referring Image Segmentation](#referring-image-segmentation)
    - [Video Captioning](#video-captioning)
    - [Embodied Agent Learning](#embodied-agent-learning)
        - [VLN (Vision-and-Language Navigation)](#vln-vision-and-language-navigation))
        - [RL (Reinforcement Learning)](#rl-reinforcement-learning))
        - [MARL (Multiagent Reinforcement Learning)](#marl-multiagent-reinforcement-learning))
    - [Egocentric Video Summarization](#egocentric-video-summarization)
    - [Hand-Object Interactions / Human Object Interaction](#hand-object-interactions)
    - [Action/Activity Recognition](#actionactivity-recognition)
    - [Action Anticipation / Gaze Anticipation](#action-anticipation)
        - [Short-Term Action Anticipation](#short-term-action-anticipation)
        - [Long-Term Action Anticipation](#long-term-action-anticipation)
        - [Future Gaze Prediction](#future-gaze-prediction)
        - [Trajectory prediction](#trajectory-prediction)
        - [Region prediction](#region-prediction)
    - [VQA (Visual Question Answering)](#vqa-visual-question-answering))
    - [VLP (Language Pre-training)](#vlp-language-pre-training)
    - [Usupervised Domain Adaptation](usupervised-domain-adaptation)
    - [Domain Generalization](domain-generalization)
    - [Multi-Modalities](#multi-modalities)
        - [Audio-Visual](#audio-visual)
        - [Depth](#depth)
- [Challenges](#challenges)



## Surveys

- [Egocentric Vision-based Action Recognition: A survey](https://www.sciencedirect.com/science/article/pii/S0925231221017586) - Adrián Núñez-Marcos, Gorka Azkune, Ignacio Arganda-Carreras, Neurocomputing 2021

- [Predicting the future from first person (egocentric) vision: A survey](https://arxiv.org/abs/2107.13411) - Ivan Rodin, Antonino Furnari, Dimitrios Mavroedis, Giovanni Maria Farinella, CVIU 2021

- [Analysis of the hands in egocentric vision: A survey](https://arxiv.org/abs/1912.10867) - Andrea Bandini, José Zariffa, TPAMI 2020

- [Summarization of Egocentric Videos: A Comprehensive Survey](https://ieeexplore.ieee.org/abstract/document/7750564) - Ana Garcia del Molino, Cheston Tan, Joo-Hwee Lim, Ah-Hwee Tan, THMS 2017

- [A survey of activity recognition in egocentric lifelogging datasets](https://ieeexplore.ieee.org/abstract/document/7934659) - El Asnaoui Khalid, Aksasse Hamid, Aksasse Brahim, Ouanan Mohammed, WITS 2017

- [Recognition of Activities of Daily Living with Egocentric Vision: A Review](https://www.mdpi.com/1424-8220/16/1/72) - Thi-Hoa-Cuc Nguyen, Jean-Christophe Nebel, Francisco Florez-Revuelta, Sensors 2016

- [The Evolution of First Person Vision Methods: A Survey](https://arxiv.org/abs/1409.1484) - Alejandro Betancourt, Pietro Morerio, Carlo S. Regazzoni, Matthias Rauterberg, TCSVT 2015

## Papers


### Episodic Memory


#### Moments Queries
- [Action Completion: A Temporal Model for Moment Detection](https://arxiv.org/pdf/1805.06749) Farnoosh Heidarivincheh, Majid Mirmehdi and Dima Damen. BMVC 2018.


- [HMDB: a large video database for human motion recognition](https://dspace.mit.edu/bitstream/handle/1721.1/69981/Poggio-HMDB.pdf?sequence=1&isAllowed=y) H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre.ICCV, 2011.


- [Beyond Action Recognition: Action Completion in RGB-D Data](https://dimadamen.github.io/ActionCompletion/ActionCompletion_BMVC2016.pdf)  Farnoosh Heidarivincheh, Majid Mirmehdi and Dima Damen. BMVC 2016.


- [With a Little Help from my Temporal Context: Multimodal Egocentric Action Recognition](https://arxiv.org/abs/2111.01024) E Kazakos, J Huh, A Nagrani, A Zisserman, D Damen. BMVC 2021. [Project](https://ekazakos.github.io/MTCN-project/) [Code](https://github.com/ekazakos/MTCN)


- [Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100](https://link.springer.com/content/pdf/10.1007/s11263-021-01531-2.pdf) D Damen, H Doughty, G Farinella, A Furnari, E Kazakos, J Ma, D Moltisanti, J Munro, T Perrett, W Price, M Wray. IJCV 2022.


- [The EPIC-KITCHENS Dataset: Collection, Challenges and Baselines.](https://arxiv.org/abs/2005.00343) 
 D Damen, H Doughty, GM Farinella, S Fidler, A Furnari, E Kazakos, D Moltisanti, J Munro, T Perrett, W Price, M Wray. IEEE Transactions on Pattern Analysis and Machine Intelligence 43(11) pp 4125-4141 (2021).



### Referring Image Segmentation

### Video Captioning 

### Embodied Agent Learning
#### VLN (Vision-and-Language Navigation)
- [Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html) Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, Anton van den Hengel. CVPR 2018. [Video](https://www.youtube.com/watch?v=Jl1NeziAHFY&list=PL_bDvITUYucCIT8iNGW8zCXeY5_u6hg-y&index=22&t=0s)

- [Vision-and-Dialog Navigation](http://proceedings.mlr.press/v100/thomason20a.html) Jesse Thomason, Michael Murray, Maya Cakmak, Luke Zettlemoyer Proceedings of the Conference on Robot Learning, PMLR 100:394-406, 2020. [Video](https://www.youtube.com/watch?v=BonlITv_PKw&feature=youtu.be)

- [Speaker-Follower Model for Vision-and-Language Navigation](https://proceedings.neurips.cc/paper/2018/hash/6a81681a7af700c6385d36577ebec359-Abstract.html) Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach, Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko, Dan Klein, Trevor Darrell. NeurIPS 2018.

- [Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Reinforced_Cross-Modal_Matching_and_Self-Supervised_Imitation_Learning_for_Vision-Language_Navigation_CVPR_2019_paper.html) Xin Wang, Qiuyuan Huang, Asli Celikyilmaz, Jianfeng Gao, Dinghan Shen, Yuan-Fang Wang, William Yang Wang, Lei Zhang. CVPR 2019

- [Learning to Navigate Unseen Environment: Back Translation with Environment Dropout](https://aclanthology.org/N19-1268.pdf) Hao Tan, Licheng Yu, Mohit Bansal, NAACL 2019 [code](https://github.com/airsplay/R2R-EnvDrop)

- [Counterfactual Vision-and-Language Navigation via Adversarial Path Sampling](https://tsujuifu.github.io/pubs/eccv20_aps.pdf) Tsu-Jui FuEmail authorXin Eric WangMatthew F. PetersonScott T. GraftonMiguel P. EcksteinWilliam Yang Wang. ECCV 2020

- [Environment-agnostic Multitask Learning for Natural Language Grounded Navigation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690409.pdf) Xin Eric WangEmail authorVihan JainEugene IeWilliam Yang WangZornitsa KozarevaSujith Ravi. ECCV 2020

- [Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation](https://arxiv.org/pdf/2007.00229) Wanrong Zhu, Xin Eric Wang, Tsu-Jui Fu, An Yan, Pradyumna Narayana, Kazoo Sone, Sugato Basu, William Yang Wang. EACL 2021.

- [Communicative Learning with Natural Gestures for Embodied Navigation Agents with Human-in-the-Scene](https://ieeexplore.ieee.org/abstract/document/9636208) -  Qi Wu, Cheng-Ju (Jimmy) Wu, Yixin Zhu, and Jungseock Joo, IROS, 2021. [code](https://github.com/qiwu57kevin/iros2021-gesthor)

#### RL (Reinforcement Learning)

- [Learning Navigation Subroutines from Egocentric Videos](http://proceedings.mlr.press/v100/kumar20a.html) - Ashish Kumar, Saurabh Gupta, Jitendra Malik,  Proceedings of the Conference on Robot Learning, PMLR 100:617-626, 2020.

- [EgoMap: Projective mapping and structured egocentric memory for Deep RL](https://chriswolfvision.github.io/www/papers/ecml2020.pdf) Edward Beeching, Christian Wolf, Jilles Dibangoye, Olivier Simonin. ECML PKDD 2020. [code](https://chriswolfvision.github.io/www/papers/ecml2020.pdf)   [video](https://crossminds.ai/video/egomap-projective-mapping-and-structured-egocentric-memory-for-deep-rl-6070b29e769086b2fca172da/) CHROMA group.

- [Deep Reinforcement Learning on a Budget: 3D Control and Reasoning Without a Supercomputer](https://ieeexplore.ieee.org/abstract/document/9412212/) Beeching, Edward; Wolf, Christian; Dibangoye, Jilles; Simonin, Olivier, CHROMA group. ICPR 2021.

- [Environment predictive coding for embodied agents Santhosh](https://arxiv.org/pdf/2102.02337.pdf)  Santhosh K. Ramakrishnan, Tushar Nagarajan, Ziad Al-Halah, Kristen Grauman. CoRR abs/2102.02337 (2021)

- [An Exploration of Embodied Visual Exploration](https://link.springer.com/content/pdf/10.1007/s11263-021-01437-z.pdf)  Santhosh K. Ramakrishnan, Dinesh Jayaraman & Kristen Grauman.


- [Shaping embodied agent behavior with activity-context priors from egocentric video](https://proceedings.neurips.cc/paper/2021/hash/f8b7aa3a0d349d9562b424160ad18612-Abstract.html) Tushar Nagarajan, Kristen Grauman. NeurIPS 2021.

- [Learning Affordance Landscapes for Interaction Exploration in 3D Environments](https://proceedings.neurips.cc/paper/2020/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf) Tushar Nagarajan, Kristen Grauman, NeurIPS 2020.


- [Explore and Explain: Self-supervised Navigation and Recounting](https://ieeexplore.ieee.org/abstract/document/9412628)  Roberto Bigazzi; Federico Landi; Marcella Cornia; Silvia Cascianelli; Lorenzo Baraldi; Rita Cucchiara. ICPR 2021

- [Embodied Visual Active Learning for Semantic Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/16338) David Nilsson, Aleksis Pirinen, Erik Gärtner, Cristian Sminchisescu, AAAI 2021.

- [IFR-Explore: Learning Inter-object Functional Relationships in 3D Indoor Scenes](https://arxiv.org/abs/2112.05298) Qi Li, Kaichun Mo, Yanchao Yang, Hang Zhao, Leonidas Guibas. Arxiv 2021.


- [Learning to Explore by Reinforcement over High-Level Options](https://arxiv.org/pdf/2111.01364) Liu Juncheng, McCane Brendan, Mills Steven. Arxiv 2021.

- [SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency](https://proceedings.neurips.cc/paper/2021/hash/6d0c932802f6953f70eb20931645fa40-Abstract.html) Devendra Singh Chaplot, Murtaza Dalal, Saurabh Gupta, Jitendra Malik, Russ R. Salakhutdinov, NeurIPS 2021.

- [Embodied Learning for Lifelong Visual Perception](https://arxiv.org/pdf/2112.14084) David Nilsson, Aleksis Pirinen, Erik Gärtner, Cristian Sminchisescu. Arxiv 2021.

- [Learning Exploration Policies for Navigation](https://arxiv.org/pdf/1903.01959) Tao Chen, Saurabh Gupta, Abhinav Gupta. ICLR 2019. [video](https://sites.google.com/view/exploration-for-nav/)

- [PLEX: PLanner and EXecutor for Embodied Learning in Navigation](https://openreview.net/pdf?id=r1g7xT4Kwr) G Avraham, Y Zuo, T Drummond.

- [RoboTHOR: An Open Simulation-to-Real Embodied AI Platform](https://openaccess.thecvf.com/content_CVPR_2020/html/Deitke_RoboTHOR_An_Open_Simulation-to-Real_Embodied_AI_Platform_CVPR_2020_paper.html)Matt Deitke, Winson Han, Alvaro Herrasti, Aniruddha Kembhavi, Eric Kolve, Roozbeh Mottaghi, Jordi Salvador, Dustin Schwenk, Eli VanderBilt, Matthew Wallingford, Luca Weihs, Mark Yatskar, Ali Farhadi. CVPR 2020. 

- [iGibson: A Simulation Environment to train Robots in Large Realistic Interactive Scenes](https://ieeexplore.ieee.org/abstract/document/9636667) Chengshu Li and Fei Xia and Roberto Mart\'in-Mart\'in and Michael Lingelbach and Sanjana Srivastava and Bokui Shen and Kent Vainio and Cem Gokmen and Gokul Dharan and Tanish Jain and Andrey Kurenkov and Karen Liu and Hyowon Gweon and Jiajun Wu and Li Fei-Fei and Silvio Savarese. [Demo](http://svl.stanford.edu/igibson/)

- [Self-Supervised Visual Reinforcement Learning with Object-Centric Representations](https://www.iclr.cc/media/Slides/iclr/2021/virtual(05-16-00)-05-16-00UTC-3331-self-supervised.pdf) Andrii Zadaianchuk, Maximilian Seitzer, Georg Martius, ICLR 2021.


#### MARL (Multiagent Reinforcement Learning)
- [A Cordial Sync: Going Beyond Marginal Policies for Multi-agent Embodied Tasks furnmove](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500460.pdf) [Code](https://github.com/allenai/cordial-sync) Jain, Unnat and Weihs, Luca and Kolve, Eric and Farhadi, Ali and Lazebnik, Svetlana and Kembhavi, Aniruddha and Schwing, Alexander G. ECCV 2020.

- [Two Body Problem: Collaborative Visual Task Completion furnlift](https://arxiv.org/pdf/1904.05879) Jain, Unnat and Weihs, Luca and Kolve, Eric and Rastegari, Mohammad and Lazebnik, Svetlana and Farhadi, Ali and Schwing, Alexander G. and Kembhavi, Aniruddha. CVPR 2019.

- [Multi-Agent Embodied Visual Semantic Navigation with Scene Prior Knowledge; Author: Xinzhu Liu](https://arxiv.org/pdf/2109.09531) Xinzhu Liu; Di Guo; Huaping Liu; Fuchun Sun; IEEE Robotics and Automation Letters, 2022.

- [Semantic Tracklets: An Object-Centric Representation for Visual Multi-Agent Reinforcement Learning](https://ioujenliu.github.io/SemanticTracklets/files/semantic_tracklets_iros21.pdf) Iou-Jen Liu; Zhongzheng Ren; Raymond A. Yeh; Alexander G. Schwing. IROS 2021.

- [Collaborative Visual Navigation](https://arxiv.org/abs/2107.01151) Haiyang Wang, Wenguan Wang, Xizhou Zhu, Jifeng Dai, Liwei Wang. Arxiv 2021.

- [Interpretation of emergent communication in heterogeneous collaborative embodied agents](https://openaccess.thecvf.com/content/ICCV2021/html/Patel_Interpretation_of_Emergent_Communication_in_Heterogeneous_Collaborative_Embodied_Agents_ICCV_2021_paper.html) Shivansh Patel, Saim Wani, Unnat Jain, Alexander G. Schwing, Svetlana Lazebnik, Manolis Savva, Angel X. Chang; ICCV 2021.

- [Agent-Centric Representations for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2104.09402) Wenling Shang, Lasse Espeholt, Anton Raichuk, Tim Salimans. Arxiv 2021.

- [GRIDTOPIX: Training Embodied Agents with Minimal Supervision](https://openaccess.thecvf.com/content/ICCV2021/html/Jain_GridToPix_Training_Embodied_Agents_With_Minimal_Supervision_ICCV_2021_paper.html) Unnat Jain, Iou-Jen Liu, Svetlana Lazebnik, Aniruddha Kembhavi, Luca Weihs, Alexander Schwing. ICCV 2021.


### Egocentric Video Summarization

- [Toward storytelling from visual lifelogging: An overview](https://arxiv.org/pdf/1507.06120.pdf) - Marc Bolanos, Mariella Dimiccoli, and Petia Radeva. In IEEE Transactions on Human-Machine Systems 2017.

- [Story-Driven Summarization for Egocentric Video](https://www.cs.utexas.edu/~grauman/papers/lu-grauman-cvpr2013.pdf) - Zheng Lu and Kristen Grauman. In CVPR 2013 [[project page]](http://vision.cs.utexas.edu/projects/egocentric/storydriven.html)

- [Discovering Important People and Objects for Egocentric Video Summarization](http://vision.cs.utexas.edu/projects/egocentric/egocentric_cvpr2012.pdf) - Yong Jae Lee, Joydeep Ghosh, and Kristen Grauman. In CVPR 2012. [[project page]](http://vision.cs.utexas.edu/projects/egocentric/index.html)


- [Video Summarization Using Deep Neural Networks: A Survey](https://ieeexplore.ieee.org/abstract/document/9594911) Evlampios Apostolidis; Eleni Adamantidou; Alexandros I. Metsai; Vasileios Mezaris; Ioannis Patras. Proceedings of the IEEE 2021.

- [Summarizing Videos with Attention](https://arxiv.org/pdf/1812.01969) Asian Conference on Computer Vision 2018. [Code](https://github.com/ok1zjf/VASNet)

- [Supervised Video Summarization via Multiple Feature Sets with Parallel Attention](https://arxiv.org/pdf/2104.11530.pdf&nbsp;&nbsp) Junaid Ahmed Ghauri; Sherzod Hakimov; Ralph Ewerth, ICME 2021. [Code](https://github.com/TIBHannover/MSVA)

- [Discriminative Feature Learning for Unsupervised Video Summarization](https://arxiv.org/pdf/1811.09791) Yunjae Jung, Donghyeon Cho, Dahun Kim, Sanghyun Woo, In So Kweon, AAAI 2018.

### Human Object Interaction

### Action/Activity Recognition

### Action/Gaze Anticipation

### VQA (Visual Question Answering) 
- [Graphhopper: Multi-Hop Scene Graph Reasoning for Visual Question Answering]( https://link.springer.com/content/pdf/10.1007%2F978-3-030-88361-4_7.pdf) - Rajat Koner, Hang Li, Marcel Hildebrandt, Deepan Das, Volker Tresp, Stephan Gunnemann ISWC 2021. [Code](https://github.com/rajatkoner08/Graphhopper)

- [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html) - Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, Devi Parikh. CVPR 2017. [Demo](https://www.youtube.com/watch?v=nMr_sSAMpkE) [Project](http://visualqa.org/)

- [Yin and Yang: Balancing and Answering Binary Visual Questions](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhang_Yin_and_Yang_CVPR_2016_paper.html) Peng Zhang, Yash Goyal, Douglas Summers-Stay, Dhruv Batra, Devi Parikh. CVPR 2016.

- [Reframing explanation as an interactive medium: The EQUAS (Explainable QUestion Answering System) project](https://onlinelibrary.wiley.com/doi/pdf/10.1002/ail2.60) William Ferguson, Dhruv Batra, Raymond Mooney, Devi Parikh, Antonio Torralba, David Bau, David Diller, Josh Fasching, Jaden Fiotto‐Kaufman, Yash Goyal, Jeff Miller, Kerry Moffitt, Alex Montes de Oca, Ramprasaath R Selvaraju, Ayush Shrivastava, Jialin Wu, Stefan Lee. Applied AI Letters 2021.

- [Question-conditioned counterfactual image generation for vqa](https://arxiv.org/pdf/1911.06352) Jingjing Pan, Yash Goyal, Stefan Lee, Arxiv 2019.

- [Towards transparent ai systems: Interpreting visual question answering models](https://arxiv.org/pdf/1608.08974) Y Goyal, A Mohapatra, D Parikh, D Batra. Arxiv 2016.

- [SOrT-ing VQA Models: Contrastive Gradient Learning for Improved Consistency](https://arxiv.org/abs/2010.10038) Sameer Dharur, Purva Tendulkar, Dhruv Batra, Devi Parikh, Ramprasaath R. Selvaraju. NAACL, 2021.

- [Contrast and Classify: Training Robust VQA Models.](https://arxiv.org/abs/2010.06087) Yash Kant, Abhinav Moudgil, Dhruv Batra, Devi Parikh, Harsh Agrawal. International Conference on Computer Vision (ICCV), 2021.

- [Dialog without Dialog Data: Learning Visual Dialog Agents from VQA Data.](https://arxiv.org/abs/2007.12750) Michael Cogswell, Jiasen Lu, Rishabh Jain, Stefan Lee, Devi Parikh, Dhruv Batra. Neural Information Processing Systems (NeurIPS), 2020.

- [Spatially Aware Multimodal Transformers for TextVQA.](https://arxiv.org/abs/2007.12146) Yash Kant, Dhruv Batra, Peter Anderson, Alex Schwing, Devi Parikh, Jiasen Lu, Harsh Agrawal. ECCV, 2020.

- [Towards VQA Models That Can Read.](https://arxiv.org/abs/1904.08920) Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, Marcus Rohrbach. CVPR, 2019. [Deom](https://textvqa.org/)



### VLP (Language Pre-training)

   - [Vision-Language Pretraining](#vision-language-pretraining)

      - [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks [NeurIPS 2019]](#vilbert-pretraining-task-agnostic-visiolinguistic-representations-for-vision-and-language-tasks-neurips-2019)
      
      - [LXMERT: Learning Cross-Modality Encoder Representations from Transformers [EMNLP 2019]](#lxmert-learning-cross-modality-encoder-representations-from-transformers-emnlp-2019)
      
      - [VisualBERT: A Simple and Performant Baseline for Vision and Language [arXiv 2019/08, ACL 2020]](#visualbert-a-simple-and-performant-baseline-for-vision-and-language-arxiv-201908-acl-2020)

      - [VL-BERT: Pre-training of Generic Visual-Linguistic Representations [ICLR 2020]](#vl-bert-pre-training-of-generic-visual-linguistic-representations-iclr-2020)

      - [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training [AAAI 2020]](#unicoder-vl-a-universal-encoder-for-vision-and-language-by-cross-modal-pre-training-aaai-2020)

      - [Unified Vision-Language Pre-Training for Image Captioning and VQA [AAAI 2020]](#unified-vision-language-pre-training-for-image-captioning-and-vqa-aaai-2020)

      - [UNITER: Learning Universal Image-text Representations [ECCV 2020]](#uniter-learning-universal-image-text-representations-eccv-2020)

      - [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks [arXiv 2020/04, ECCV 2020]](#oscar-object-semantics-aligned-pre-training-for-vision-language-tasks-arxiv-202004-eccv-2020)

      - [Learning Transferable Visual Models From Natural Language Supervision [OpenAI papers 2021/01]](#learning-transferable-visual-models-from-natural-language-supervision-openai-papers-202101)
    
  - [Video-Language Pretraining](#video-language-pretraining)
  
      - [VideoBERT: A Joint Model for Video and Language Representation Learning [ICCV 2019]](#videobert-a-joint-model-for-video-and-language-representation-learning-iccv-2019)
    
      - [Multi-modal Transformer for Video Retrieval [ECCV 2020]](#multi-modal-transformer-for-video-retrieval-eccv-2020)
    
      - [HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training [EMNLP 2020]](#hero-hierarchical-encoder-for-videolanguage-omni-representation-pre-training-emnlp-2020)
    
      - [UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation](#univl-a-unified-video-and-language-pre-training-model-for-multimodal-understanding-and-generation)




### Action/Activity Recognition

- [Stacked Temporal Attention: Improving First-person Action Recognition by Emphasizing Discriminative Clips](https://arxiv.org/abs/2112.01038) - Lijin Yang, Yifei Huang, Yusuke Sugano, Yoichi Sato, BMVC 2021

- [With a Little Help from my Temporal Context: Multimodal Egocentric Action Recognition](https://arxiv.org/abs/2111.01024) - Evangelos Kazakos, Jaesung Huh, Arsha Nagrani, Andrew Zisserman, Dima Damen, BMVC 2021

- [Interactive Prototype Learning for Egocentric Action Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Interactive_Prototype_Learning_for_Egocentric_Action_Recognition_ICCV_2021_paper.html) Xiaohan Wang, Linchao Zhu, Heng Wang, Yi Yang, ICCV 2021.

- [Learning to Recognize Actions on Objects in Egocentric Video with Attention Dictionaries](https://ieeexplore.ieee.org/abstract/document/9353268) - Swathikiran Sudhakaran, Sergio Escalera, Oswald Lanz, T-PAMI 2021

- [Slow-Fast Auditory Streams For Audio Recognition](https://arxiv.org/abs/2103.03516) - Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen, ICASSP 2021

- [Integrating Human Gaze Into Attention for Egocentric Activity Recognition](https://openaccess.thecvf.com/content/WACV2021/html/Min_Integrating_Human_Gaze_Into_Attention_for_Egocentric_Activity_Recognition_WACV_2021_paper.html) - Kyle Min, Jason J. Corso, WACV 2021.

- [Self-Supervised Joint Encoding of Motion and Appearance for First Person Action Recognition](https://arxiv.org/pdf/2002.03982.pdf) - Mirco Planamente, Andrea Bottino, Barbara Caputo, ICPR 2020

- [Gate-Shift Networks for Video Action Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Sudhakaran_Gate-Shift_Networks_for_Video_Action_Recognition_CVPR_2020_paper.html) - Swathikiran Sudhakaran, Sergio Escalera, Oswald Lanz, CVPR 2020. [[code]](https://github.com/swathikirans/GSM)

- [Trear: Transformer-based RGB-D Egocentric Action Recognition](https://ieeexplore.ieee.org/abstract/document/9312201?casa_token=VjrXPrZDuSgAAAAA:ezQgxMoeH7q3fxl8su7zg1yghkp60nbxCwU3FxyZEKWghbUVozmKmS_YE99AYceBr3lxA6Ud) - Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, Wanqing Li, TCDS 2020

- [EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kazakos_EPIC-Fusion_Audio-Visual_Temporal_Binding_for_Egocentric_Action_Recognition_ICCV_2019_paper.pdf) - Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima, ICCV 2019. [[code]](https://github.com/ekazakos/temporal-binding-network) [[project page]](https://ekazakos.github.io/TBN/)

- [Learning Spatiotemporal Attention for Egocentric Action Recognition](http://openaccess.thecvf.com/content_ICCVW_2019/papers/EPIC/Lu_Learning_Spatiotemporal_Attention_for_Egocentric_Action_Recognition_ICCVW_2019_paper.pdf) - Minlong Lu, Danping Liao, Ze-Nian Li, WICCV 2019

- [Multitask Learning to Improve Egocentric Action Recognition](https://arxiv.org/abs/1909.06761) - Georgios Kapidis, Ronald Poppe, Elsbeth van Dam, Lucas Noldus, Remco Veltkamp, WICCV 2019

- [Seeing and Hearing Egocentric Actions: How Much Can We Learn?](https://arxiv.org/abs/1910.06693) - Alejandro Cartas, Jordi Luque, Petia Radeva, Carlos Segura, Mariella Dimiccoli, WICCV19

- [Deep Attention Network for Egocentric Action Recognition](https://ieeexplore.ieee.org/abstract/document/8653357) - Minlong Lu, Simon Fraser, Ze-Nian Li, Yueming Wang, Gang Pan, TIP 2019

- [LSTA: Long Short-Term Attention for Egocentric Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sudhakaran_LSTA_Long_Short-Term_Attention_for_Egocentric_Action_Recognition_CVPR_2019_paper.pdf) - Sudhakaran, Swathikiran and Escalera, Sergio and Lanz, Oswald, CVPR 2019. [[code]](https://github.com/swathikirans/LSTA)

- [Long-Term Feature Banks for Detailed Video Understanding](https://arxiv.org/abs/1812.05038) - Chao-Yuan Wu, Christoph Feichtenhofer, Haoqi Fan, Kaiming He, Philipp Krähenbühl, Ross Girshick, CVPR 2019

- [Attention is All We Need: Nailing Down Object-centric Attention for Egocentric Activity Recognition](https://arxiv.org/abs/1807.11794) - Swathikiran Sudhakaran, Oswald Lanz, BMVC 2018 

- [Egocentric Activity Recognition on a Budget](https://openaccess.thecvf.com/content_cvpr_2018/papers/Possas_Egocentric_Activity_Recognition_CVPR_2018_paper.pdf) - Possas, Rafael and Caceres, Sheila Pinto and Ramos, Fabio, CVPR 2018. [[demo]](https://youtu.be/GBo4sFNzhtU)

- [In the eye of beholder: Joint learning of gaze and actions in first person video](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yin_Li_In_the_Eye_ECCV_2018_paper.pdf) - Li, Y., Liu, M., & Rehg, J. M., ECCV 2018.

- [Egocentric Gesture Recognition Using Recurrent 3D Convolutional Neural Networks with Spatiotemporal Transformer Modules](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_Egocentric_Gesture_Recognition_ICCV_2017_paper.pdf) - Cao, Congqi and Zhang, Yifan and Wu, Yi and Lu, Hanqing and Cheng, Jian, ICCV 2017.

- [Action recognition in RGB-D egocentric videos](https://ieeexplore.ieee.org/document/8296915) - Yansong Tang, Yi Tian, Jiwen Lu, Jianjiang Feng, Jie Zhou, ICIP 2017

- [Trajectory Aligned Features For First Person Action Recognition](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2016/Suriya_2016_Trajectory_Features.pdf) - S. Singh, C. Arora, and C.V. Jawahar, Pattern Recognition 2017.

- [Modeling Sub-Event Dynamics in First-Person Action Recognition](https://openaccess.thecvf.com/content_cvpr_2017/html/Zaki_Modeling_Sub-Event_Dynamics_CVPR_2017_paper.html) - Hasan F. M. Zaki, Faisal Shafait, Ajmal Mian, CVPR 2017

- [First Person Action Recognition Using Deep Learned Descriptors](https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S12-15.pdf) - S. Singh, C. Arora, and C.V. Jawahar, CVPR 2016. [[project page]](http://cvit.iiit.ac.in/research/projects/cvit-projects/first-person-action-recognition) [[code]](https://github.com/suriyasingh/EgoConvNet)

- [Delving into egocentric actions](https://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Delving_Into_Egocentric_2015_CVPR_paper.pdf) - Li, Y., Ye, Z., & Rehg, J. M., CVPR 2015.

- [Pooled Motion Features for First-Person Videos](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ryoo_Pooled_Motion_Features_2015_CVPR_paper.pdf) - Michael S. Ryoo, Brandon Rothrock and Larry H. Matthies, CVPR 2015.

- [Generating Notifications for Missing Actions: Don't forget to turn the lights off!](https://homes.cs.washington.edu/~ali/alarm-iccv.pdf) - Soran, Bilge, Ali Farhadi, and Linda Shapiro, ICCV 2015.

- [First-Person Activity Recognition: What Are They Doing to Me?](http://cvrc.ece.utexas.edu/mryoo/papers/cvpr2013_ryoo.pdf) - M. S. Ryoo and L. Matthies, CVPR 2013.

- [Detecting activities of daily living in first-person camera views](https://www.cs.cmu.edu/~deva/papers/ADL_2012.pdf) - Pirsiavash, H., & Ramanan, D., CVPR 2012.

- [Learning to recognize daily actions using gaze](http://ai.stanford.edu/~alireza/publication/ECCV12.pdf) - Fathi, A., Li, Y., & Rehg, J. M, ECCV 2012.


#### Hand-Object Interactions

- [Learning Visual Affordance Grounding from Demonstration Videos](https://arxiv.org/abs/2108.05675) - Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao, 2021

- [Domain and View-point Agnostic Hand Action Recognition](https://arxiv.org/abs/2103.02303) - Alberto Sabater, Iñigo Alonso, Luis Montesano, Ana C. Murillo, 2021

- [Understanding Egocentric Hand-Object Interactions from Hand Estimation](https://arxiv.org/abs/2109.14657) - Yao Lu, Walterio W. Mayol-Cuevas, 2021

- [Egocentric Hand-object Interaction Detection and Application](https://arxiv.org/abs/2109.14734) - Yao Lu, Walterio W. Mayol-Cuevas, 2021

- [The MECCANO Dataset: Understanding Human-Object Interactions from Egocentric Videos in an Industrial-like Domain](https://arxiv.org/abs/2010.05654) - Francesco Ragusa and Antonino Furnari and Salvatore Livatino and Giovanni Maria Farinella, WACV 2021. [[project page]](https://iplab.dmi.unict.it/MECCANO/)

- [Is First Person Vision Challenging for Object Tracking?](https://arxiv.org/abs/2011.12263) - Matteo Dunnhofer, Antonino Furnari, Giovanni Maria Farinella, Christian Micheloni, WICCV 2021

- [Real Time Egocentric Object Segmentation: THU-READ Labeling and Benchmarking Results](https://arxiv.org/abs/2106.04957) - E. Gonzalez-Sosa, G. Robledo, D. Gonzalez-Morin, P. Perez-Garcia, A. Villegas, WCVPR 2021

- [Forecasting Human-Object Interaction: Joint Prediction of Motor Attention and Actions in First Person Video](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460681.pdf) - Miao Liu, Siyu Tang, Yin Li, James M. Rehg, ECCV 2020. [[project page]](https://aptx4869lm.github.io/ForecastingHOI/)

- [Understanding Human Hands in Contact at Internet Scale](https://openaccess.thecvf.com/content_CVPR_2020/html/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.html)  - Dandan Shan, Jiaqi Geng, Michelle Shu, David F. Fouhey, CVPR 2020

- [Generalizing Hand Segmentation in Egocentric Videos with Uncertainty-Guided Model Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cai_Generalizing_Hand_Segmentation_in_Egocentric_Videos_With_Uncertainty-Guided_Model_Adaptation_CVPR_2020_paper.pdf) - Minjie Cai and Feng Lu and Yoichi Sato, CVPR 2020. [[code]](https://github.com/cai-mj/UMA)

- [Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kulon_Weakly-Supervised_Mesh-Convolutional_Hand_Reconstruction_in_the_Wild_CVPR_2020_paper.pdf) - Dominik Kulon, Riza Alp Güler, Iasonas Kokkinos, Michael Bronstein, Stefanos Zafeiriou, CVPR 2020 

- [Hand-Priming in Object Localization for Assistive Egocentric Vision](https://openaccess.thecvf.com/content_WACV_2020/papers/Lee_Hand-Priming_in_Object_Localization_for_Assistive_Egocentric_Vision_WACV_2020_paper.pdf) - Lee, Kyungjun and Shrivastava, Abhinav and Kacorri, Hernisa, WACV 2020.

- [Learning joint reconstruction of hands and manipulated objects](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hasson_Learning_Joint_Reconstruction_of_Hands_and_Manipulated_Objects_CVPR_2019_paper.pdf) - Yana Hasson, Gül Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, Cordelia Schmid, CVPR 2020

- [H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tekin_HO_Unified_Egocentric_Recognition_of_3D_Hand-Object_Poses_and_Interactions_CVPR_2019_paper.pdf) - Tekin, Bugra and Bogo, Federica and Pollefeys, Marc, CVPR 2019. [[video]](https://youtu.be/ko6kNZ9DuAk?t=3240)

- [From Lifestyle VLOGs to Everyday Interaction](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0733.pdf) - David F. Fouhey and Weicheng Kuo and Alexei A. Efros and Jitendra Malik, CVPR 2018. [[project page]](http://web.eecs.umich.edu/~fouhey/2017/VLOG/index.html)

- [Analysis of Hand Segmentation in the Wild](https://arxiv.org/pdf/1803.03317) - Aisha Urooj, Ali Borj, CVPR 2018.

- [First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations](https://openaccess.thecvf.com/content_cvpr_2018/papers/Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf) - Garcia-Hernando, Guillermo and Yuan, Shanxin and Baek, Seungryul and Kim, Tae-Kyun, CVPR 2018. [[project page]](https://guiggh.github.io/publications/first-person-hands/) [[code]](https://github.com/guiggh/hand_pose_action)

- [Jointly Recognizing Object Fluents and Tasks in Egocentric Videos](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Jointly_Recognizing_Object_ICCV_2017_paper.pdf) - Liu, Yang and Wei, Ping and Zhu, Song-Chun, ICCV 2017.

- [Egocentric Gesture Recognition Using Recurrent 3D Convolutional Neural Networks with Spatiotemporal Transformer Modules](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_Egocentric_Gesture_Recognition_ICCV_2017_paper.pdf) - Cao, Congqi and Zhang, Yifan and Wu, Yi and Lu, Hanqing and Cheng, Jian, ICCV 2017.

- [First Person Action-Object Detection with EgoNet](https://arxiv.org/abs/1603.04908) - Gedas Bertasius, Hyun Soo Park, Stella X. Yu, Jianbo Shi, 2017

- [Understanding Hand-Object Manipulation with Grasp Types and Object Attributes](http://www.cs.cmu.edu/~kkitani/pdf/CKY-RSS16.pdf) - Minjie Cai and Kris M. Kitani and Yoichi Sato, Robotics: Science and Systems 2016.

- [Lending a hand: Detecting hands and recognizing activities in complex egocentric interactions](http://homes.sice.indiana.edu/sbambach/papers/iccv-egohands.pdf) - Bambach, S., Lee, S., Crandall, D. J., & Yu, C., ICCV 2015.

- [Understanding Everyday Hands in Action From RGB-D Images](https://openaccess.thecvf.com/content_iccv_2015/html/Rogez_Understanding_Everyday_Hands_ICCV_2015_paper.html) - Gregory Rogez, James S. Supancic III, Deva Ramanan, ICCV 2015

- [You-Do, I-Learn: Discovering Task Relevant Objects and their Modes of Interaction from Multi-User Egocentric Video](http://www.bmva.org/bmvc/2014/files/paper059.pdf) - Dima Damen, Teesid Leelasawassuk, Osian Haines, Andrew Calway, and Walterio Mayol-Cuevas, BMVC 2014

- [Detecting Snap Points in Egocentric Video with a Web Photo Prior](https://www.cs.utexas.edu/~grauman/papers/bo-eccv2014.pdf) - Bo Xiong and Kristen Grauman, ECCV 2014. [[project page]](http://vision.cs.utexas.edu/projects/ego_snappoints/) [[code]](http://vision.cs.utexas.edu/projects/ego_snappoints/#code)

- [3D Hand Pose Detection in Egocentric RGB-D Images](https://link.springer.com/chapter/10.1007/978-3-319-16178-5_25) - Grégory Rogez, Maryam Khademi, J. S. Supančič III, J. M. M. Montiel, Deva Ramanan, WECCV 2014

- [Pixel-level hand detection in ego-centric videos](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Li_Pixel-Level_Hand_Detection_2013_CVPR_paper.pdf) - Li, Cheng, and Kris M. Kitani. CVPR 2013. [[video]](https://youtu.be/N756YmLpZyY) [[code]](https://github.com/irllabs/handtrack)

- [Learning to recognize objects in egocentric activities](https://homes.cs.washington.edu/~xren/publication/fathi_cvpr11_egocentric_objects.pdf) - Fathi, A., Ren, X., & Rehg, J. M., CVPR 2011.

- [Context-based vision system for place and object recognition](https://www.cs.ubc.ca/~murphyk/Papers/iccv03.pdf) - Torralba, A., Murphy, K. P., Freeman, W. T., & Rubin, M. A., ICCV 2003. [[project page]](https://www.cs.ubc.ca/~murphyk/Vision/placeRecognition.html)


#### Usupervised Domain Adaptation

- [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://arxiv.org/abs/2110.10101) - Mirco Planamente, Chiara Plizzari, Emanuele Alberti, Barbara Caputo, WACV 2022

- [Differentiated Learning for Multi-Modal Domain Adaptation](https://dl.acm.org/doi/pdf/10.1145/3474085.3475660?casa_token=wOh7PYXIrGoAAAAA:WBP-sajm70r9KKNqNcwM7RIMW9D_re7MC56V10yq3_GCh4JafS_JegifZJ8--87l5TEcucuGaTYM) - Jianming Lv, Kaijie Liu, Shengfeng He, MM 2021

- [Domain Adaptation in Multi-View Embedding for Cross-Modal Video Retrieval](https://arxiv.org/abs/2110.12812) - Jonathan Munro, Michael Wray, Diane Larlus, Gabriela Csurka, Dima Damen, 2021

- [Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing](https://openreview.net/forum?id=a1wQOh27zcy) - Aadarsh Sahoo, Rutav Shah, Rameswar Panda, Kate Saenko, Abir Das, NIPS 2021

- [Learning Cross-modal Contrastive Features for Video Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Learning_Cross-Modal_Contrastive_Features_for_Video_Domain_Adaptation_ICCV_2021_paper.pdf) - Donghyun Kim, Yi-Hsuan Tsai, Bingbing Zhuang, Xiang Yu, Stan Sclaroff, Kate Saenko, Manmohan Chandraker, ICCV 2021

- [Spatio-temporal Contrastive Domain Adaptation for Action Recognition](https://openaccess.thecvf.com/content/CVPR2021/html/Song_Spatio-temporal_Contrastive_Domain_Adaptation_for_Action_Recognition_CVPR_2021_paper.html) - Xiaolin Song, Sicheng Zhao, Jingyu Yang, Huanjing Yue, Pengfei Xu, Runbo Hu, Hua Chai, CVPR 2021

- [Multi-Modal Domain Adaptation for Fine-Grained Action Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Munro_Multi-Modal_Domain_Adaptation_for_Fine-Grained_Action_Recognition_CVPR_2020_paper.html) - Jonathan Munro, Dima Damen, CVPR 2020 

#### Domain Generalization

- [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://arxiv.org/abs/2110.10101) - Mirco Planamente, Chiara Plizzari, Emanuele Alberti, Barbara Caputo, WACV 2022

### Action Anticipation

#### Short-Term Action Anticipation

- [Action Anticipation Using Pairwise Human-Object Interactions and Transformers](https://ieeexplore.ieee.org/abstract/document/9546623?casa_token=S642phYzKBsAAAAA:N3U0R4Hj7qiWztApTVHFJitkK8zFux5RTqTzjE6fRaT8luL4gVW-l-Fzoqd-K4u0x0bHvGgpjPI) - Debaditya Roy; Basura Fernando, TIP 2021

- [Higher Order Recurrent Space-Time Transformer for Video Action Prediction](https://arxiv.org/abs/2104.08665) - Tsung-Ming Tai, Giuseppe Fiameni, Cheng-Kuang Lee, Oswald Lanz, ArXiv 2021

- [Anticipating Human Actions by Correlating Past With the Future With Jaccard Similarity Measures](https://openaccess.thecvf.com/content/CVPR2021/html/Fernando_Anticipating_Human_Actions_by_Correlating_Past_With_the_Future_With_CVPR_2021_paper.html) - Basura Fernando, Samitha Herath, CVPR 2021

- [Towards Streaming Egocentric Action Anticipation](https://arxiv.org/abs/2110.05386) - Antonino Furnari, Giovanni Maria Farinella, arXiv 2021

- [Multimodal Global Relation Knowledge Distillation for Egocentric Action Anticipation](https://dl.acm.org/doi/abs/10.1145/3474085.3475327?casa_token=S_eM0ZcL9G8AAAAA:n9-Xa3-WzAD5NGx9h9WA7ZnBFW5Xzv-QYu-wWYtUaqpYAagALI37qL1rc3WWawgiNf_0VrtOWX0) - Y Huang, X Yang, C Xu, ACM 2021

- [Multi-Modal Temporal Convolutional Network for Anticipating Actions in Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2021W/Precognition/html/Zatsarynna_Multi-Modal_Temporal_Convolutional_Network_for_Anticipating_Actions_in_Egocentric_Videos_CVPRW_2021_paper.html) - Olga Zatsarynna, Yazan Abu Farha, Juergen Gall, CVPRW 2021

- [Self-Regulated Learning for Egocentric Video Activity Anticipation](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) - Zhaobo Qi; Shuhui Wang; Chi Su; Li Su; Qingming Huang; Qi Tian, T-PAMI 2021

- [Anticipative Video Transformer](https://arxiv.org/abs/2106.02036) - Rohit Girdhar, Kristen Grauman, ICCV 2021

- [What If We Could Not See? Counterfactual Analysis for Egocentric Action Anticipation](ijcai.org/proceedings/2021/182) - T Zhang, W Min, J Yang, T Liu, S Jiang, Y Rui, IJCAI 2021 

- [Rolling-Unrolling LSTMs for Action Anticipation from First-Person Video](https://arxiv.org/abs/2005.02190) - Antonino Furnari, Giovanni Maria Farinella, T-PAMI 2020

- [Knowledge Distillation for Action Anticipation via Label Smoothing](https://arxiv.org/abs/2004.07711) - Guglielmo Camporese, Pasquale Coscia, Antonino Furnari, Giovanni Maria Farinella, Lamberto Ballan, ICPR 2020

- [An Egocentric Action Anticipation Framework via Fusing Intuition and Analysis](https://dl.acm.org/doi/10.1145/3394171.3413964) - Tianyu Zhang, Weiqing Min, Ying Zhu, Yong Rui, Shuqiang Jiang, ACM 2020

- [What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention](https://arxiv.org/pdf/1905.09035) - Antonino Furnari, Giovanni Maria Farinella, ICCV 2019 [[code]](https://github.com/fpv-iplab/rulstm) [[demo]](https://youtu.be/buIEKFHTVIg)

- [Forecasting Human-Object Interaction: Joint Prediction of Motor Attention and Actions in First Person Video](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460681.pdf) - Miao Liu, Siyu Tang, Yin Li, James M. Rehg, ECCV 2020. [[project page]](https://aptx4869lm.github.io/ForecastingHOI/)

- [Leveraging the Present to Anticipate the Future in Videos](https://arxiv.org/abs/2004.07711) - Antoine Miech, Ivan Laptev, Josef Sivic, Heng Wang, Lorenzo Torresani, Du Tran, CVPRW 2019

- [Zero-Shot Anticipation for Instructional Activities](https://ieeexplore.ieee.org/document/9008304) - Fadime Sener, Angela Yao, ICCV 2019


#### Long-Term Action Anticipation

- [Learning to Anticipate Egocentric Actions by Imagination](https://arxiv.org/pdf/2101.04924.pdf) - Yu Wu, Linchao Zhu, Xiaohan Wang, Yi Yang, Fei Wu, TIP 2021. 

- [On Diverse Asynchronous Activity Anticipation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740766.pdf) - He Zhao and Richard P. Wildes, ECCV 2020

- [Time-Conditioned Action Anticipation in One Shot](https://openaccess.thecvf.com/content_CVPR_2019/html/Ke_Time-Conditioned_Action_Anticipation_in_One_Shot_CVPR_2019_paper.html) - Qiuhong Ke, Mario Fritz, Bernt Schiele, CVPR 2019

- [When Will You Do What? - Anticipating Temporal Occurrences of Activities](https://openaccess.thecvf.com/content_cvpr_2018/html/Abu_Farha_When_Will_You_CVPR_2018_paper.html) - Yazan Abu Farha, Alexander Richard, Juergen Gall, CVPR 2018

- [Joint Prediction of Activity Labels and Starting Times in Untrimmed Videos](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mahmud_Joint_Prediction_of_ICCV_2017_paper.pdf) - Tahmida Mahmud, Mahmudul Hasan, Amit K. Roy-Chowdhury, ICCV 2017

- [First-Person Activity Forecasting with Online Inverse Reinforcement Learning](https://arxiv.org/pdf/1612.07796) - Nicholas Rhinehart, Kris M. Kitani, ICCV 2017. [[project page]](https://people.eecs.berkeley.edu/~nrhinehart/darko.html) [[video]](https://youtu.be/rvVoW3iuq-s) 

#### Future Gaze Prediction

- [Unsupervised gaze prediction in egocentric videos by energy-based surprise modeling](http://arxiv.org/abs/2001.11580), Aakur, S.N., Bagavathi, A., ArXiv 2020

- [Digging Deeper into Egocentric Gaze Prediction](https://arxiv.org/pdf/1904.06090) - Hamed R. Tavakoli and Esa Rahtu and Juho Kannala and Ali Borji, WACV 2019.

- [Predicting Gaze in Egocentric Video by Learning Task-dependent Attention Transition](https://arxiv.org/pdf/1803.09125) - Huang, Y., Cai, M., Li, Z., & Sato, Y., ECCV 2018 [[code]](https://github.com/hyf015/egocentric-gaze-prediction)

- [Deep future gaze: Gaze anticipation on egocentric videos using adversarial networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Deep_Future_Gaze_CVPR_2017_paper.pdf) - Zhang, M., Teck Ma, K., Hwee Lim, J., Zhao, Q., & Feng, J., CVPR 2017. [[code]](https://github.com/Mengmi/deepfuturegaze_gan)

- [Learning to predict gaze in egocentric video](http://ai.stanford.edu/~alireza/publication/Li-Fathi-Rehg-ICCV13.pdf) - Li, Yin, Alireza Fathi, and James M. Rehg, ICCV 2013.


#### Trajectory prediction

- [Forecasting Action through Contact Representations from First Person Video](https://ieeexplore.ieee.org/abstract/document/9340014?casa_token=PUk2a8mN4CoAAAAA:ICkziPRIBtlxgzsyJm9ZVxUIzGnEq0phTHLOP8G8TxFlTIp159calFp8jZOdUCnxeWTknFjlB0w) - Eadom Dessalene; Chinmaya Devaraj; Michael Maynord; Cornelia Fermuller; Yiannis Aloimonos, T-PAMI 2021

- [Multimodal Future Localization and Emergence Prediction for Objects in Egocentric View With a Reachability Prior](https://openaccess.thecvf.com/content_CVPR_2020/papers/Makansi_Multimodal_Future_Localization_and_Emergence_Prediction_for_Objects_in_Egocentric_CVPR_2020_paper.pdf) - Makansi, Osama and Cicek, Ozgun and Buchicchio, Kevin and Brox, Thomas, CVPR 2020. [[demo]](https://youtu.be/_9Ml5IFwbSY) [[code]](https://github.com/lmb-freiburg/FLN-EPN-RPN) [[project page]](https://lmb.informatik.uni-freiburg.de/Publications/2020/MCBB20/)

- [Understanding Human Hands in Contact at Internet Scale](https://openaccess.thecvf.com/content_CVPR_2020/html/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.html) - Dandan Shan, Jiaqi Geng, Michelle Shu, David F. Fouhey, CVPR 2020

- [Forecasting Human-Object Interaction: Joint Prediction of Motor Attention and Actions in First Person Video](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460681.pdf) - Miao Liu, Siyu Tang, Yin Li, James M. Rehg, ECCV 2020. [[project page]](https://aptx4869lm.github.io/ForecastingHOI/)

- [How Can I See My Future? FvTraj: Using First-person View for Pedestrian Trajectory Prediction](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520562.pdf) - Huikun Bi, Ruisi Zhang, Tianlu Mao, Zhigang Deng, Zhaoqi Wang, ECCV 2020. [[presentation video]](https://youtu.be/HcsyH7zMHAw) [[summary video]](https://youtu.be/X1cSNWT6Gr0)

- [Future Person Localization in First-Person Videos](https://ieeexplore.ieee.org/document/8578890)- Takuma Yagi; Karttikeya Mangalam; Ryo Yonetani; Yoichi Sato, CVPR 2018

- [Egocentric Future Localization](https://openaccess.thecvf.com/content_cvpr_2016/papers/Park_Egocentric_Future_Localization_CVPR_2016_paper.pdf) - Park, Hyun Soo and Hwang, Jyh-Jing and Niu, Yedong and Shi, Jianbo, CVPR 2016. [[demo]](https://youtu.be/i_9CTMZ60zc)

- [Going deeper into first-person activity recognition](http://www.cs.cmu.edu/~kkitani/pdf/MFK-CVPR2016.pdf) - Ma, M., Fan, H., & Kitani, K. M., CVPR 2016.


#### Region prediction

- [EGO-TOPO: Environment Affordances from Egocentric Video](https://openaccess.thecvf.com/content_CVPR_2020/papers/Nagarajan_Ego-Topo_Environment_Affordances_From_Egocentric_Video_CVPR_2020_paper.pdf) - Nagarajan, Tushar and Li, Yanghao and Feichtenhofer, Christoph and Grauman, Kristen, CVPR 2020. [[project page]](http://vision.cs.utexas.edu/projects/ego-topo/) [[demo]](http://vision.cs.utexas.edu/projects/ego-topo/demo.html)

- [ Forecasting human object interaction: Joint prediction of motor attention and egocentric activity](http://arxiv.org/abs/1911.10967) - Liu, M., Tang, S., Li, Y., Rehg, J., arXiv 2019

- [Forecasting Hands and Objects in Future Frames](https://openaccess.thecvf.com/content_eccv_2018_workshops/w15/html/Fan_Forecasting_Hands_and_Objects_in_Future_Frames_ECCVW_2018_paper.html) - Chenyou Fan, Jangwon Lee, Michael S. Ryoo, ECCVW 2018

- [Next-active-object prediction from egocentric videos](https://www.sciencedirect.com/science/article/abs/pii/S1047320317301967) - Antonino Furnari, Sebastiano Battiato, Kristen Grauman, Giovanni Maria Farinella, JVCIR 2017

- [First Person Action-Object Detection with EgoNet](https://arxiv.org/abs/1603.04908), G Bertasius, HS Park, SX Yu, J Shi, arXiv 2016

- [Unsupervised Learning of Important Objects From First-Person Videos](https://openaccess.thecvf.com/content_iccv_2017/html/Bertasius_Unsupervised_Learning_of_ICCV_2017_paper.html) - Gedas Bertasius, Hyun Soo Park, Stella X. Yu, Jianbo Shi, ICCV 2017



### Multi-Modalities

#### Audio-Visual

- [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135), Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun, NIPS 2021

- [Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition](https://arxiv.org/abs/2110.10101) - Mirco Planamente, Chiara Plizzari, Emanuele Alberti, Barbara Caputo, WACV 2022

- [With a Little Help from my Temporal Context: Multimodal Egocentric Action Recognition](https://arxiv.org/abs/2111.01024) - Evangelos Kazakos, Jaesung Huh, Arsha Nagrani, Andrew Zisserman, Dima Damen, BMVC 2021

- [Slow-Fast Auditory Streams For Audio Recognition](https://arxiv.org/abs/2103.03516) - Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen, ICASSP 2021

- [Multi-modal Egocentric Activity Recognition using Audio-Visual Features](https://arxiv.org/pdf/1807.00612.pdf) - Mehmet Ali Arabacı, Fatih Özkan, Elif Surer, Peter Jančovič, Alptekin Temizel, MTA 2020

- [EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kazakos_EPIC-Fusion_Audio-Visual_Temporal_Binding_for_Egocentric_Action_Recognition_ICCV_2019_paper.pdf) - Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima, ICCV 2019. [[code]](https://github.com/ekazakos/temporal-binding-network) [[project page]](https://ekazakos.github.io/TBN/)

- [Seeing and Hearing Egocentric Actions: How Much Can We Learn?](https://arxiv.org/abs/1910.06693) - Alejandro Cartas, Jordi Luque, Petia Radeva, Carlos Segura, Mariella Dimiccoli, WICCV19


#### Depth

- [Trear: Transformer-based RGB-D Egocentric Action Recognition](https://ieeexplore.ieee.org/abstract/document/9312201?casa_token=VjrXPrZDuSgAAAAA:ezQgxMoeH7q3fxl8su7zg1yghkp60nbxCwU3FxyZEKWghbUVozmKmS_YE99AYceBr3lxA6Ud) - Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, Wanqing Li, TCDS 2020

- [First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations](https://openaccess.thecvf.com/content_cvpr_2018/papers/Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf) - Garcia-Hernando, Guillermo and Yuan, Shanxin and Baek, Seungryul and Kim, Tae-Kyun, CVPR 2018. [[project page]](https://guiggh.github.io/publications/first-person-hands/) [[code]](https://github.com/guiggh/hand_pose_action)

- [Multi-stream Deep Neural Networks for RGB-D Egocentric Action Recognition](http://www.cs.toronto.edu/~zianwang/MDNN/TCSVT18_MDNN.pdf) - Yansong Tang, Zian Wang, Jiwen Lu, Jianjiang Feng, Jie Zhou, TCSVT 2018

- [Action recognition in RGB-D egocentric videos](https://ieeexplore.ieee.org/document/8296915) - Yansong Tang, Yi Tian, Jiwen Lu, Jianjiang Feng, Jie Zhou, ICIP 2017

- [Scene Semantic Reconstruction from Egocentric RGB-D-Thermal Videos](https://ieeexplore.ieee.org/abstract/document/8374614) - Rachel Luo, Ozan Sener, Silvio Savarese, 3DV 2017

- [3D Hand Pose Detection in Egocentric RGB-D Images](https://link.springer.com/chapter/10.1007/978-3-319-16178-5_25) - Grégory Rogez, Maryam Khademi, J. S. Supančič III, J. M. M. Montiel, Deva Ramanan, WECCV 2014

#### Thermal

- [Scene Semantic Reconstruction from Egocentric RGB-D-Thermal Videos](https://ieeexplore.ieee.org/abstract/document/8374614) - Rachel Luo, Ozan Sener, Silvio Savarese, 3DV 2017

#### Event 

- [E(GO)^2MOTION: Motion Augmented Event Stream for Egocentric Action Recognition](https://arxiv.org/abs/2112.03596) - Chiara Plizzari, Mirco Planamente, Gabriele Goletto, Marco Cannici, Emanuele Gusso, Matteo Matteucci, Barbara Caputo, 2021

### Temporal Segmentation (Action Detection)

- [UnweaveNet: Unweaving Activity Stories](https://arxiv.org/pdf/2112.10194.pdf) - Will Price, Carl Vondrick, Dima Damen, 2021

- [Temporal Action Segmentation from Timestamp Supervision](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Temporal_Action_Segmentation_From_Timestamp_Supervision_CVPR_2021_paper.pdf) - Zhe Li, Yazan Abu Farha, Jurgen Gall, CVPR 2021

- [Personal-Location-Based Temporal Segmentation of Egocentric Video for Lifelogging Applications](https://iplab.dmi.unict.it/PersonalLocationSegmentation/downloads/furnari2018personal.pdf) - A. Furnari, G. M. Farinella, S. Battiato, Journal of Visual Communication and Image Representation 2017 [[demo]](https://youtu.be/URM0EdYuKEw) [[project page]](https://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/)

- [Temporal segmentation and activity classification from first-person sensing](https://ieeexplore.ieee.org/document/5204354) - Spriggs, Ekaterina H., Fernando De La Torre, and Martial Hebert, Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2009.

### Retrieval 

- [Domain Adaptation in Multi-View Embedding for Cross-Modal Video Retrieval](https://arxiv.org/abs/2110.12812) - Jonathan Munro, Michael Wray, Diane Larlus, Gabriela Csurka, Dima Damen, 2021

- [On Semantic Similarity in Video Retrieval](https://openaccess.thecvf.com/content/CVPR2021/papers/Wray_On_Semantic_Similarity_in_Video_Retrieval_CVPR_2021_paper.pdf) - Michael Wray, Hazel Doughty, Dima Damen, CVPR 2021

- [Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf) - Michael Wray, Diane Larlus, Gabriela Csurka, Dima Damen, ICCV 2019

### Few-Shot Action Recognition

- [Unifying Few- and Zero-Shot Egocentric Action Recognition](https://arxiv.org/abs/2006.11393) - Tyler R. Scott, Michael Shvartsman, Karl Ridgeway, CVPRW 2021


### Gaze

- [1000 Pupil Segmentations in a Second Using Haar Like Features and Statistical Learning](https://openaccess.thecvf.com/content/ICCV2021W/EPIC/html/Fuhl_1000_Pupil_Segmentations_in_a_Second_Using_Haar_Like_Features_ICCVW_2021_paper.html) - Wolfgang Fuhl, Johannes Schneider, Enkelejda Kasneci, WICCV 2021

### From Third-Person to First-Person 

- [Ego-Exo: Transferring Visual Representations From Third-Person to First-Person Videos]() - Yanghao Li, Tushar Nagarajan, Bo Xiong, Kristen Grauman, CVPR 2021

- [Actor and Observer: Joint Modeling of First and Third-Person Videos](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sigurdsson_Actor_and_Observer_CVPR_2018_paper.pdf) - Gunnar A. Sigurdsson and Abhinav Gupta and Cordelia Schmid and Ali Farhadi and Karteek Alahari, CVPR 2018. [[code]](https://github.com/gsig/actor-observer)

- [Making Third Person Techniques Recognize First-Person Actions in Egocentric Videos](https://ieeexplore.ieee.org/abstract/document/8451249?casa_token=p1k79yrTIkMAAAAA:QHlXMC8Y7qrCEDsdypGNZbh7zeoEPVEs2k6j5a0g1MkvA76Uf6_VDIfCzbiG2bWdU8EoFyagbK4) - Sagar Verma, Pravin Nagar, Divam Gupta, Chetan Arora, ICIP 2018



### User Data from an Egocentric Point of View

- [Dynamics-regulated kinematic policy for egocentric pose estimation](https://proceedings.neurips.cc/paper/2021/hash/d1fe173d08e959397adf34b1d77e88d7-Abstract.html) - Zhengyi Luo, Ryo Hachiuma, Ye Yuan, Kris Kitani, NIPS 2021

- [Estimating Egocentric 3D Human Pose in Global Space](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Estimating_Egocentric_3D_Human_Pose_in_Global_Space_ICCV_2021_paper.html) - Jian Wang, Lingjie Liu, Weipeng Xu, Kripasindhu Sarkar, Christian Theobalt, ICCV 2021

- [Egocentric Pose Estimation From Human Vision Span](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Egocentric_Pose_Estimation_From_Human_Vision_Span_ICCV_2021_paper.html) - Hao Jiang, Vamsi Krishna Ithapu, ICCV 2021

- [EgoRenderer: Rendering Human Avatars From Egocentric Camera Images](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_EgoRenderer_Rendering_Human_Avatars_From_Egocentric_Camera_Images_ICCV_2021_paper.html) - Tao Hu, Kripasindhu Sarkar, Lingjie Liu, Matthias Zwicker, Christian Theobalt, ICCV 2021

- [Whose Hand Is This? Person Identification From Egocentric Hand Gestures](https://openaccess.thecvf.com/content/WACV2021/html/Tsutsui_Whose_Hand_Is_This_Person_Identification_From_Egocentric_Hand_Gestures_WACV_2021_paper.html) - Satoshi Tsutsui, Yanwei Fu, David J. Crandall, WACV 2021.

- [Recognizing Camera Wearer from Hand Gestures in Egocentric Videos](https://dl.acm.org/doi/pdf/10.1145/3394171.3413654?casa_token=tlspOQU5qekAAAAA:rM0hbyyg1cvY5KRK16blErILxTO_OJpU9CIr8W9nDxBbdvjJBNxyKJ5GcNWTjrgJwV_H_Me8cFlj) - Daksh Thapar, Aditya Nigam, Chetan Arora, MM 2020, [code](https://egocentricbiometric.github.io/) 

- [You2Me: Inferring Body Pose in Egocentric Video via First and Second Person Interactions](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ng_You2Me_Inferring_Body_Pose_in_Egocentric_Video_via_First_and_CVPR_2020_paper.pdf) - Ng, Evonne and Xiang, Donglai and Joo, Hanbyul and Grauman, Kristen, CVPR 2020. [[demo]](http://vision.cs.utexas.edu/projects/you2me/demo.mp4) [[project page]](http://vision.cs.utexas.edu/projects/you2me/) [[dataset]](https://github.com/facebookresearch/you2me/tree/master/data#) [[code]](https://github.com/facebookresearch/you2me#)

- [Ego-Pose Estimation and Forecasting as Real-Time PD Control](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yuan_Ego-Pose_Estimation_and_Forecasting_As_Real-Time_PD_Control_ICCV_2019_paper.pdf) - Ye Yuan and Kris Kitani, ICCV 2019. [[code]](https://github.com/Khrylx/EgoPose) [[project page]](https://www.ye-yuan.com/ego-pose) [[demo]](https://youtu.be/968IIDZeWE0)

- [xR-EgoPose: Egocentric 3D Human Pose From an HMD Camera](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tome_xR-EgoPose_Egocentric_3D_Human_Pose_From_an_HMD_Camera_ICCV_2019_paper.pdf) - Tome, Denis and Peluse, Patrick and Agapito, Lourdes and Badino, Hernan, ICCV 2019. [[demo]](https://youtu.be/zem03fZWLrQ) [[dataset]](https://github.com/facebookresearch/xR-EgoPose)

- [3D Ego-Pose Estimation via Imitation Learning](https://openaccess.thecvf.com/content_ECCV_2018/html/Ye_Yuan_3D_Ego-Pose_Estimation_ECCV_2018_paper.html) - Ye Yuan, Kris Kitani, ECCV 2018

### Localization 

- [Egocentric Indoor Localization From Room Layouts and Image Outer Corners](https://openaccess.thecvf.com/content/ICCV2021W/EPIC/html/Chen_Egocentric_Indoor_Localization_From_Room_Layouts_and_Image_Outer_Corners_ICCVW_2021_paper.html) - Xiaowei Chen, Guoliang Fan, WICCV 2021

- [Egocentric Activity Recognition and Localization on a 3D Map](https://arxiv.org/abs/2105.09544) - Miao Liu, Lingni Ma, Kiran Somasundaram, Yin Li, Kristen Grauman, James M. Rehg, Chao Li, 2021

- [Egocentric Shopping Cart Localization](https://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/home/_paper/egocentric%20shopping%20cart%20localization.pdf) - E. Spera, A. Furnari, S. Battiato, G. M. Farinella, ICPR 2018.

- [Recognizing personal locations from egocentric videos](https://ieeexplore.ieee.org/document/7588113) - Furnari, A., Farinella, G. M., & Battiato, S., IEEE Transactions on Human-Machine Systems 2017.

- [Context-based vision system for place and object recognition](https://www.cs.ubc.ca/~murphyk/Papers/iccv03.pdf) - Torralba, A., Murphy, K. P., Freeman, W. T., & Rubin, M. A., ICCV 2003. [[project page]](https://www.cs.ubc.ca/~murphyk/Vision/placeRecognition.html)

### Privacy protection

- [Anonymizing Egocentric Videos](https://openaccess.thecvf.com/content/ICCV2021/papers/Thapar_Anonymizing_Egocentric_Videos_ICCV_2021_paper.pdf) - Daksh Thapar, Aditya Nigam, Chetan Arora, ICCV  2021

- [Mitigating Bystander Privacy Concerns in Egocentric Activity Recognition with Deep Learning and Intentional Image Degradation](http://users.ece.utexas.edu/~ethomaz/papers/j2.pdf) - Dimiccoli, M., Marín, J., & Thomaz, E., Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 2018.

- [Privacy-Preserving Human Activity Recognition from Extreme Low Resolution](https://arxiv.org/pdf/1604.03196) - Ryoo, M. S., Rothrock, B., Fleming, C., & Yang, H. J., AAAI 2017.

### Social Interactions

- [EgoCom: A Multi-person Multi-modal Egocentric Communications Dataset](https://ieeexplore.ieee.org/document/9200754) - Curtis G. Northcutt and Shengxin Zha and Steven Lovegrove and Richard Newcombe, PAMI 2020.

- [Deep Dual Relation Modeling for Egocentric Interaction Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Deep_Dual_Relation_Modeling_for_Egocentric_Interaction_Recognition_CVPR_2019_paper.pdf) - Li, Haoxin and Cai, Yijun and Zheng, Wei-Shi, CVPR 2019.

- [Recognizing Micro-Actions and Reactions from Paired Egocentric Videos](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yonetani_Recognizing_Micro-Actions_and_CVPR_2016_paper.pdf) - Yonetani, Ryo and Kitani, Kris M. and Sato, Yoichi, CVPR 2016.

- [Social interactions: A first-person perspective](http://www.cs.utexas.edu/~cv-fall2012/slides/jake-expt.pdf) - Fathi, A., Hodgins, J. K., & Rehg, J. M., CVPR 2012.


### Multiple Egocentric Tasks

- [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058) - Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Christian Fuegen, Abrham Gebreselasie, Cristina Gonzalez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Kolar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo, Yuchen Wang, Xindi Wu, Takuma Yagi, Yunyi Zhu, Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria Farinella, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, Jitendra Malik, arXiv. [[Github]](https://github.com/EGO4D) [[project page]](https://ego4d-data.org) [[video]](https://drive.google.com/file/d/1oknfQIH9w1rXy6I1j5eUE6Cqh96UwZ4L/view?usp=sharing)

### Activity-context

- [Learning Visual Affordance Grounding from Demonstration Videos](https://arxiv.org/abs/2108.05675) - Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao, 2021

- [Shaping embodied agent behavior with activity-context priors from egocentric video](https://proceedings.neurips.cc/paper/2021/file/f8b7aa3a0d349d9562b424160ad18612-Paper.pdf) - Tushar Nagarajan, Kristen Grauman, NIPS 2021

- [EGO-TOPO: Environment Affordances from Egocentric Video](https://openaccess.thecvf.com/content_CVPR_2020/html/Nagarajan_Ego-Topo_Environment_Affordances_From_Egocentric_Video_CVPR_2020_paper.html) - Tushar Nagarajan, Yanghao Li, Christoph Feichtenhofer, Kristen Grauman, CVPR 2020


### Video summarization

- [Egocentric video summarisation via purpose-oriented frame scoring and selection](https://www.sciencedirect.com/science/article/pii/S0957417421014159) - V. Javier Traver and Dima Damen, Expert Systems with Applications 2022

- [Together Recognizing, Localizing and Summarizing Actions in Egocentric Videos](https://ieeexplore.ieee.org/abstract/document/9399266?casa_token=R8LKJM45-MgAAAAA:Pfjxjt8k7l4SD_iopfL9JYsq2k6ShpZGATkXg-z5B4BTuPQV3A4HqhtZ2VqhVPtiIVbPIi_oaPU) - Abhimanyu Sahu; Ananda S. Chowdhury, TIP 2021

- [First person video summarization using different graph representations](https://www.sciencedirect.com/science/article/pii/S0167865521001008?casa_token=H7rMpQAduAsAAAAA:Aq6ryy4IihojkZ9Tj3LoYRxT66VO3KmdBIiRTJoDvd_WBNIsHJxhruPTSNrzR6NniRg8iYyk) - Abhimanyu Sahu, Ananda S.Chowdhury, Pattern Recognition Letters 2021

- [Text Synopsis Generation for Egocentric Videos](https://ieeexplore.ieee.org/abstract/document/9412111?casa_token=Vf3uXoASupsAAAAA:CHg-misMhCLcZn-CWdUFFBLJ_SGlvsmZrAc-lfujd5yxVQSF0pr13RAYSdmrOTfYaTB0xKTj_Wg) - Aidean Sharghi; Niels da Vitoria Lobo; Mubarak Shah, ICPR 2020

- [Personalized Egocentric Video Summarization of Cultural Tour on User Preferences Input](https://ieeexplore.ieee.org/abstract/document/7931584?casa_token=gsvjlImpjQQAAAAA:F420NFVZd0V3igjGLVv8VpXnD1Ul5SakMxlfwcdAYCwNTsEjPgrLAKMhnUKX2VgOpoJgRm03XzI) - Patrizia Varini; Giuseppe Serra; Rita Cucchiara, IEEE Transactions on Multimedia 2017

- [Highlight Detection with Pairwise Deep Ranking for First-Person Video Summarization](https://ieeexplore.ieee.org/document/7780481) - Ting Yao; Tao Mei; Yong Rui, CVPR 2016

- [Video Summarization with Long Short-term Memory](https://arxiv.org/abs/1605.08110) - Ke Zhang, Wei-Lun Chao, Fei Sha, Kristen Grauman, ECCV 2016

- [Discovering Picturesque Highlights from Egocentric Vacation Videos](https://arxiv.org/abs/1601.04406) - Vinay Bettadapura, Daniel Castro, Irfan Essa, arXiv 2016 

- [Spatial and temporal scoring for egocentric video summarization](https://www.sciencedirect.com/science/article/pii/S0925231216304805?casa_token=2uf2ekbvb7cAAAAA:YxtgDl8G6D-uunhYGOGv_aMgJeWefuO9klkQdMIh-jXz3V4JzEocy_Og3pPbaWMIlG2URM5t) - Zhao Guo, Lianli Gao, Xiantong Zhen, Fuhao Zou, Fumin Shen, Kai Zheng, Neurocomputing 2016

- [Gaze-Enabled Egocentric Video Summarization via Constrained Submodular Maximization](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Xu_Gaze-Enabled_Egocentric_Video_2015_CVPR_paper.html) - Jia Xu, Lopamudra Mukherjee, Yin Li, Jamieson Warner, James M. Rehg, Vikas Singh, CVPR 2015

- [Predicting Important Objects for Egocentric Video Summarization](https://link.springer.com/article/10.1007/s11263-014-0794-5?sa_campaign=email/event/articleAuthor/onlineFirst&error=cookies_not_supported&error=cookies_not_supported&code=9f9cd56d-eec9-49eb-bb9f-229724e371da&code=a2d596a3-5527-4ece-addc-1db7b036c200) - Yong Jae Lee & Kristen Grauman, IJCV 2015

- [Video Summarization by Learning Submodular Mixtures of Objectives](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gygli_Video_Summarization_by_2015_CVPR_paper.pdf) - Michael Gygli, Helmut Grabner, Luc Van Gool, CVPR 2015

- [Storyline Representation of Egocentric Videos with an Applications to Story-Based Search](https://ieeexplore.ieee.org/document/7410871) - Bo Xiong; Gunhee Kim; Leonid Sigal, ICCV 2015

- [Detecting Snap Points in Egocentric Video with a Web Photo Prior](https://www.cs.utexas.edu/~grauman/papers/bo-eccv2014.pdf) - Bo Xiong and Kristen Grauman, ECCV 2014

- [Creating Summaries from User Videos](https://gyglim.github.io/me/papers/GygliECCV14_vsum.pdf) - Michael Gygli, Helmut Grabner, Hayko Riemenschneider, and Luc Van Gool, ECCV 2014

- [Quasi Real-Time Summarization for Consumer Videos](https://www.cs.cmu.edu/~epxing/papers/2014/Zhao_Xing_cvpr14a.pdf) - Bin Zhao,  Eric P. Xing, CVPR 2014

- [Story-Driven Summarization for Egocentric Video](https://www.cs.utexas.edu/~grauman/papers/lu-grauman-cvpr2013.pdf) - Zheng Lu and Kristen Grauman, CVPR 2013 [[project page]](http://vision.cs.utexas.edu/projects/egocentric/storydriven.html)

- [Discovering Important People and Objects for Egocentric Video Summarization](http://vision.cs.utexas.edu/projects/egocentric/egocentric_cvpr2012.pdf) - Yong Jae Lee, Joydeep Ghosh, and Kristen Grauman, CVPR 2012. [[project page]](http://vision.cs.utexas.edu/projects/egocentric/index.html)

- [Wearable hand activity recognition for event summarization](https://ieeexplore.ieee.org/document/1550796) - Mayol, W. W., & Murray, D. W., IEEE International Symposium on Wearable Computers, 2005.

### Applications

- [Wearable System for Personalized and Privacy-preserving Egocentric Visual Context Detection using On-device Deep Learning](https://dl.acm.org/doi/abs/10.1145/3450614.3461684) - Mina Khan, Glenn Fernandes, Akash Vaish, Mayank Manuja, Pattie Maes, UMAP 2021

- [Learning Robot Activities From First-Person Human Videos Using Convolutional Future Regression](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/Lee_Learning_Robot_Activities_CVPR_2017_paper.html) - Jangwon Lee, Michael S. Ryoo, CVPR 2017

### Human to Robot

- [Learning Robot Activities From First-Person Human Videos Using Convolutional Future Regression](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/Lee_Learning_Robot_Activities_CVPR_2017_paper.html) - Jangwon Lee, Michael S. Ryoo, CVPR 2017

- [One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning](http://www.roboticsproceedings.org/rss14/p02.pdf) - Tianhe Yu, Chelsea Finn, Annie Xie, Sudeep Dasari, Tianhao Zhang, Pieter Abbeel, Sergey Levine, RSS 2014


### Asssitive Egocentric Vision

- [A Computational Model of Early Word Learning from the Infant's Point of View](https://arxiv.org/abs/2006.02802) - Satoshi Tsutsui, Arjun Chandrasekaran, Md Alimoor Reza, David Crandall, Chen Yu, CogSci 2020

- [Preserved action recognition in children with autism spectrum disorders: Evidence from an EEG and eye-tracking study](https://onlinelibrary.wiley.com/doi/10.1111/psyp.13740) - Mohammad Saber Sotoodeh, Hamidreza Taheri-Torbati, Nouchine Hadjikhani, Amandine Lassalle, Psychophysiology 2020

### Popular Architectures

#### 2D

- [GSM] [Gate-Shift Networks for Video Action Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Sudhakaran_Gate-Shift_Networks_for_Video_Action_Recognition_CVPR_2020_paper.html) - Swathikiran Sudhakaran, Sergio Escalera, Oswald Lanz, CVPR 2020. [[code]](https://github.com/swathikirans/GSM)
- [TSM] [TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html) - Ji Lin, Chuang Gan, Song Han, ICCV 2019 
- [TBN] [EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kazakos_EPIC-Fusion_Audio-Visual_Temporal_Binding_for_Egocentric_Action_Recognition_ICCV_2019_paper.pdf) - Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima, ICCV 2019. [[code]](https://github.com/ekazakos/temporal-binding-network) [[project page]](https://ekazakos.github.io/TBN/)
- [TRN] [Temporal Relational Reasoning in Videos](https://arxiv.org/pdf/1711.08496.pdf) - Bolei Zhou, Alex Andonian, Aude Oliva, Antonio Torralba, ECCV 2018. [[project page]](http://relation.csail.mit.edu/)
- [R(2+1)] [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html) - Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri, CVPR 2018
- [TSN] [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2) - Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool, ECCV 2016

#### 3D

- [SlowFast] [SlowFast Networks for Video Recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html) - Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He, ICCV 2019
- [I3D] [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html) - Joao Carreira, Andrew Zisserman, CVPR 2017

#### RNN

- [LSTA] [LSTA: Long Short-Term Attention for Egocentric Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sudhakaran_LSTA_Long_Short-Term_Attention_for_Egocentric_Action_Recognition_CVPR_2019_paper.pdf) - Sudhakaran, Swathikiran and Escalera, Sergio and Lanz, Oswald, CVPR 2019. [[code]](https://github.com/swathikirans/LSTA)
- [RULSTM] [What Would You Expect? Anticipating Egocentric Actions with Rolling-Unrolling LSTMs and Modality Attention](https://arxiv.org/pdf/1905.09035) - Antonino Furnari, Giovanni Maria Farinella, ICCV 2019 [[code]](https://github.com/fpv-iplab/rulstm) [[demo]](https://youtu.be/buIEKFHTVIg)

#### Transformer 

- [XViT] [Space-time Mixing Attention for Video Transformer](https://proceedings.neurips.cc/paper/2021/file/a34bacf839b923770b2c360eefa26748-Paper.pdf) - Adrian Bulat, Juan-Manuel Perez-Rua, Swathikiran Sudhakaran, Brais Martinez, Georgios Tzimiropoulos, NIPS 2021
- [ViViT] [ViViT: A Video Vision Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.html) Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid, ICCV 2021
- [TimeSformer] [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) - Gedas Bertasius, Heng Wang, Lorenzo Torresani, ICML 2021

### Other EGO-Context

- [Revisiting 3D Object Detection From an Egocentric Perspective](https://proceedings.neurips.cc/paper/2021/hash/db182d2552835bec774847e06406bfa2-Abstract.html) - Boyang Deng, Charles R. Qi, Mahyar Najibi, Thomas Funkhouser, Yin Zhou, Dragomir Anguelov, NIPS 2021

- [Learning by Watching](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_by_Watching_CVPR_2021_paper.pdf) - Jimuyang Zhang, Eshed Ohn-Bar, CVPR 2021



## Challenges

- [Ego4D](https://ego4d-data.org) - Episodic Memory, Hand-Object Interactions, AV Diarization, Social, Forecasting.

- [Epic Kithchen Challenge](https://epic-kitchens.github.io/2021) - Action Recognition, Action Detection, Action Anticipation, Unsupervised Domain Adaptation for Action Recognition, Multi-Instance Retrieval
