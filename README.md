# VERTE-X-Aging
Code for "Spine age estimation using deep learning in lateral spine radiographs and DXA VFA to predict incident fracture and mortality". 

## Estimate spine age
In this study, we developed a convolutional neural network model to estimate spine age from lateral spine radiographs and dual energy X-ray absorptiometry (DXA) vertebral fracture assessment (VFA) images. Discriminatory performance for prevalent vertebral fracture and osteoporosis was compared for biological spine age versus chronological age. Prognostic value of predicted spine age difference for incident fracture and mortality was assessed with adjustment for chronological age, sex, and covariates.

<p align="center" width="100%">
    <img width="100%" src="./imgs/prediction_Res.png"> 
    <em>Associations of chronological age and predicted spine age in the (A) derivation test set (spine radiograph cohort, aged 40 years or older n=2063) and (B) external test set (DXA VFA cohort, age 65 years or older, n=3508).</em>
</p>


## Image processing

Because of the intensity difference in individual images, histogram equalization was applied to all images, and Min-Max scaling method was selected to normalize the pixel intensity values of the images. About 5% of the image size was cropped for areas not related to analysis, and the cropped images were resized according to the size (1024, 512). Since the width and height differed by each image, we set the width to 512 px if the width was larger than the height, then resized the height according to the resolution ratio. If an image was smaller than (1024,512), the image was aligned to the center position with zero-padding to the rest of the area. As VFA images had a narrower lateral width with focus on the thoracic and lumbar spine, no cropping was needed for the VFA images. 

## Mean-variance loss used in the spine age prediction model

While other studies addressing age prediction used exact age regression models, we found that exact regression does not effectively leverage the robustness of distributions in capturing labels with inherent ambiguity. Given that both X-ray and VFA images are grayscale, limiting their expressiveness and diversity compared to RGB images, it was essential to incorporate distributional analysis. To address this, we applied the mean-variance loss function proposed by Pan (2019).

*Pan, Hongyu et al. “Mean-Variance Loss for Deep Age Estimation from a Face.” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018): 5285-5294.([paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.pdf))*


<p align="center" width="100%">
    <img width="100%" src="./imgs/lossfunction.jpg"> 
    <em>Deep learning model architecture to predict spine age</em>
</p>


The mean loss penalizes the difference between the mean of an estimated spine age distribution and the ground-truth age. Different from softmax loss which focuses on classification tasks, our mean loss emphasizes on regression tasks, and we use the L2 distance to measure the distance between the mean of an estimated age distribution and the ground-truth age. Therefore, it is complementary to the softmax loss. Such a variance loss requires that an estimated distribution should be concentrated at a small range of the mean. The variance loss penalizes the dispersion of an estimated spine age distribution, making it as sharp as possible. This is helpful to obtain an accurate spine age estimation with a narrow confidence interval.

We applied this mean-variance loss into the architecture of convolutional neural network model, and the softmax loss and mean-variance loss was used jointly as the supervision signal. The final Loss of the spine age prediction model could be represented as:
$$L_final=L_c+λ_1 L_m+λ_2 L_v$$
where $λ_1$ and $λ_2$ are two hyper-parameters, balancing the influencing of individual sub-losses in the joint loss. Initially, $λ_1$ and $λ_2$ was 0.2 and 0.05, respectively. 
In the inference phase, the age of a test image is estimated as:
$$y_p=r(\displaystyle\sum_{i=40}^{K} i*p_i)$$
where $p_i$, $i∈{40,41,42,…,K}$ is the output of the softmax layer in the network, and $r(∙)$ is a round function.

## Age-level bias correction between different modalities

To reduce modality-specific differences, an age-level bias correction was applied to the external VFA test set based on Beheshti’s method.

*Zhang B, Zhang S, Feng J, Zhang S. Age-level bias correction in brain age prediction. NeuroImage Clinical 2023; 37: 103319. ([paper](https://www.sciencedirect.com/science/article/pii/S2213158223000086?via%3Dihub))*

*Beheshti I, Nugent S, Potvin O, Duchesne S. Bias-adjustment in neuroimaging-based brain age frameworks: A robust scheme. NeuroImage Clinical 2019; 24: 102063. ([paper](https://www.sciencedirect.com/science/article/pii/S2213158219304103?via%3Dihub))*



