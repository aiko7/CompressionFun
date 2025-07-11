First row is the original images, second row is the reconstructed version of the images from a latent vector compressed down to 4% of the size of the original image.
<img width="1200" height="300" alt="image" src="https://github.com/user-attachments/assets/692d6572-9100-4867-9abc-466d58fdd356" />
Important to remember that the reconstructed image itself isnt smalll so it's not a traditional compression method, 
instead the latent space in which the vector is stored before reconstruction is very small however that vector + the
decoder is everything you need to reconstruct the image whenever needed so it's still an effective approach
