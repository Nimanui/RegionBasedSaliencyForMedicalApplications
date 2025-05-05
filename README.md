# Region Based Saliency For Medical Applications
Repository for exploring applying region based saliency mapping to arbitrary deep learning applications with a medical use

The goal here is to apply saliency mapping to a deep learning model applied to visual medical data. From there we apply region mapping to see if we can improve the use and readability 
of the saliency mapping in a statistically significant way. The premise is that saliency mapping is useful for medical professionals to understand how deep learning models are making decisions,
and that the understanding of those models could be improved with region mapping, particularly in the case of deep learning models applied to difficult applications.

The original premise and research for this is presented in the paper "Region-based Saliency Explanations on the Recognition of Facial Genetic Syndromes". As access to the custom dataset utilized in the paper could not be acquired, I instead used an arbitrary CNN applied to chest x-ray data, and attempted to apply the core premise and elements of the paper to that dataset and deep learning model instead.
Conveniently I was able to do this by extending a lab we did in the class "Deep Learning for Healthcare" at UIUC.

# Saliency Mapping Functions
In the paper 15 different saliency mapping functions were utilized. For simplicity and due to time constraints I picked out what looked to be the most successful, and visually interesting functions and applied those. In retrospect the paper did use a library called "Zenith" to apply the saliency functions, which if I had taken that approach may have saved me some time and allowed me to apply more functions than the 3 I was able to apply in the time available.

# Gradient
This appears to be among the most basic options and is available pretty much out-of-the-box from pytorch. I ultimately applied this one first and used it more as a proof of concept for how the saliency mapping works. To add gradient saliency, you create a dataloader and set the "requires_grad" property on the pytorch tensor. You can then feed this into your trained model, during which pytorch will generate a gradient (or weight) tensor for how the model views the image. From there it is a simple matter to extract that tensor and display it with matplotlib.

![image](https://github.com/user-attachments/assets/8a9d9d6d-ee62-42ee-a809-c46ea4e71bb0)

# GradCAM
This is a more sophisticated application of gradient saliency that uses hooks while the model is running apply relu and interpolation to specific layers while you feed data through the model. This produces a down-sampled saliency map that can then be resized and displayed on top of the original image to show the saliency.

![image](https://github.com/user-attachments/assets/fc397ed4-9298-45fd-b079-197511d1d0f7)

# LRP-ε-Z⁺ Rule
Like with GradCAM LRP leverages hooks to observe specific layers, but goes a step further to compute a relevance score for each neuron applying a non-trivial summation function.
For each neuron $i$ and output neuron $j$, the relevance score is computed as:

$$
R_i = \sum_j \frac{z_{ij}^+}{\sum_i z_{ij}^+ + \epsilon \cdot \text{sign}\left(\sum_i z_{ij}^+\right)} R_j
$$

Where:
- $ z_{ij}^+ = x_i w_{ij}^+ $ : the product of the input and **positive part** of the weight.
- $R_j$ : relevance at layer $j$
- $ \epsilon $ : small positive constant for numerical stability

![image](https://github.com/user-attachments/assets/c33575b8-c5db-44bc-abe8-cf4691daa11e)

# Visualization
From there we look into fun ways to interactively visualize the saliency images on top of the existing input image, both as a single image and in bulk batches. I had some success with using sliders that allow you to shift dynamically between the base image and the saliency image to clarify where the model is focusing.

# Region based Saliency
For the final step we look at the original premise of the paper and attempt to apply region mapping to try to reduce the noise in the saliency mapping relative to known relevant regions. Unlike the paper, which had specific small regions that could then be applied using facial recognition algorithms, I opted for a more broad approach as the relevant parts of the chest xray (the lungs in the case of the Pneumonia dataset), take up more of the space of the images. I had some hope this would be enough even without an algorithm to adjust the regions to better align with the images, but found that any of the more generic region breakdowns I tried were largely not statistically significant, even when applying a more generous alpha than the paper did.

I ran out of time to dig in deeper in this area, but generally the next step would be to try to better adjust the regions to actually align with the lungs and see if that is more statistically significant as the paper indicated. Further more saliency mapping algorithms could be implemented and all of this could be applied against other deep learning models.

Sümer, Ö., Waikel, R.L., Hanchard, S.E.L., Duong, D., Krawitz, P., Conati, C., Solomon, B.D. &amp; André, E.. (2023). Region-based Saliency Explanations on the Recognition of Facial Genetic Syndromes. <i>Proceedings of the 8th Machine Learning for Healthcare Conference</i>, in <i>Proceedings of Machine Learning Research</i> 219:712-736 Available from https://proceedings.mlr.press/v219/sumer23a.html.
