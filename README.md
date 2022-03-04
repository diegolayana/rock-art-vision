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
    * Resizing. ✅
    * First moment of area. _(getting the center)_ ✅
    * Second moment of area. _(getting the orientation)_ ✅
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
![](images/images_readme/image_raw.png)
As you can see its a really easy image to work with. If you were asking why, its because its a *binary image*, it means we are only working with one channel (array or matrix) instead of three (the case of colored image).
#### Thresholding
For us to know which should be the range of values for this kind of image, we should extract a *histogram*, something like the spectrum of this image.
```python
img = Imagen(r'C:\\...\images_raw\img.tif')
plt.hist(img.img.ravel(),256,[0,256])
plt.show
```
![](images/images_readme/hist.png)

