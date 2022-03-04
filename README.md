# Rock Art Vision

Computer vision 👁️ program for Rock Art analysis. 

## Introduction

_This program is part of my work to obtain Electronic Engieneering degree._ 🤕

I'm this work, a methodology for the orgaization and analisys of rock art data from Arica y Parinacota region is being proposed. 
Workig with drawings from this artifeacts a curvature analisys is being development for a automatic classification.
Also, manim animations is beign studied for visualization porpuses during the final presentation.

## Progress
The following list is the composition of the code and its progress: 

_this list is going to be updated everyday (i wish)_

Simbology: ✅ = Ready, ☑️ = In progress, ⌚ = In waiting

* Automatic Classification: ☑️
  * Preprocessing: ✅
    * Thresholding. ✅
    * First moment of area. _(getting the center)_ ✅
    * Second moment of area. _(getting the orientation)_ ✅
    * Resizing. ✅
    * Initialization. ✅

  * Segmentation: ☑️
    * Control points.✅
    * Contour lines.✅
      * Greedy algorithm. ✅
      * Active contour model.☑️
    * Initialization. ⌚

  * Curvature Scale Space mapping (CSS-map): ☑️
    * Contour initializacion. ☑️
    * -the rest of the code- ⌚

  * Support Vector Machine algorithm (SVM): ⌚
  
* Visualization: ⌚
_manim animation for presenting the final work_
## Some code.
_for you to belive in my programming habilities_
### Preprocessing

Here im gonna show some functions from `imagen.py` file

We are starting with this drawing, an interpretation from a rock art.
![](images/images_readme/image_raw.png)__
As you can see its a really easy image to work with. If you were asking why, its because its a *binary image*, it means we are only working with one channel (array or matrix) instead of three (the case of colored image).
#### Thresholding
For us to know which should be the range of values for this kind of image, we should extract as *histogram*, something like the spectrum of this image.
```python
img = Imagen(r'C:\\...\images_raw\img.tif')
plt.hist(img.img.ravel(),256,[0,256])
plt.show
```
![](images/images_readme/hist.png)__
This is how we should be hable to know where to apply the threshold.
Applying 
```python
thresh = img.thresh()
plot(thresh)
```
![](images/images_readme/image_preprocessed.png)__

#### First moment of area 
This is a real important feature for us to positioning the object at the center of the image.
The function was coded following the next formula.__
<img src="https://latex.codecogs.com/svg.image?S_x&space;=&space;A\bar{y}&space;=&space;\Sigma_{i=1}^{n}y_i&space;d&space;A_i&space;=&space;\int_{A}ydA" title="S_x = A\bar{y} = \Sigma_{i=1}^{n}y_i d A_i = \int_{A}ydA" />__
and __
<img src="https://latex.codecogs.com/svg.image?S_y&space;=&space;A\bar{x}&space;=&space;\Sigma_{i=1}^{n}x_i&space;d&space;A_i&space;=&space;\int_{A}xdA" title="S_y = A\bar{x} = \Sigma_{i=1}^{n}x_i d A_i = \int_{A}xdA" />__

```python
fm = img.first_moment()
plot(fm)
```
![](images/images_readme/first_moment.png)__

#### Second momento of area.
![](images/images_readme/second_moment.png)__

### Segmentation.
First we define a vector of point for us to start the contour algorithm
![](images/images_readme/segmentation_init.png)__

### Greedy alhorithm.

In this section the a greedy approach is used for the control points to fit the contour of the images. For us to do that, first we defined a energy image, gived by the Blurred Gradient Magnitude Squared.__

<img src="https://latex.codecogs.com/svg.image?||\Delta&space;n_{\sigma}*I||^{2}" title="||\Delta n_{\sigma}*I||^{2}" />__

![](images/images_readme/blurgrad.png)__

The algorithm is supossed to search for the maximum of the evergy funtion. In a mathematics language:__

<img src="https://latex.codecogs.com/svg.image?\equiv&space;Max[&space;E_{image}]\\\equiv&space;Max[\Sigma_{i=0}^{n-1}||\Delta&space;n_{\sigma}*I(v_i)||^2]]" title="\equiv Max[ E_{image}]\\\equiv Max[\Sigma_{i=0}^{n-1}||\Delta n_{\sigma}*I(v_i)||^2]]" />__

Currently the function `def contour()` are giving the next results;__

![](images/images_readme/contour.png)__

So we have to add some constrains and start reaching the optimal contour for the CSS-maping
