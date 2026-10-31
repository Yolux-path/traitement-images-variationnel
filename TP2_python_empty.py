import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.signal import convolve2d


# ## 1

# On paper

# ## 2

# On paper

# # Denoising with variational approaches




def scalar_product(u, v):
    if u.ndim > 2:
        return np.sum(u * v, axis=0)
    else:
        return np.sum(u * v)

def norm(u):
    return np.sqrt(scalar_product(u, u))





def gradient(u):
    m, n = u.shape
    grad_u = np.zeros((2, m, n))
    
    grad_u[0, :-1, :] = u[1:] - u[:-1]
    
    grad_u[1, :, :-1] = u[:, 1:] - u[:, :-1]
    
    return grad_u





def div(p):
    m, n = p.shape[1:]
    
    div_1 = np.zeros((m, n))
    div_1[:-1, :] = p[0, :-1, :]
    div_1[1:, :] -= p[0, :-1, :]
    
    div_2 = np.zeros((m, n))
    div_2[:, :-1] = p[1, :, :-1]
    div_2[:, 1:] -= p[1, :, :-1]
    
    return div_1 + div_2





def laplacian(u):
    return div(gradient(u))


# #### Image setup




def load_img(fname, dirname='images/'):
    Im_PIL = Image.open(dirname + fname)
    return np.array(Im_PIL)





def add_gaussian_noise(I, s):
    m, n = I.shape
    
    I_out = I + s * np.random.randn(m, n)
    
    return I_out





# Load image
Im_data = load_img('cameraman.png')





# Noise image
Im_noised = add_gaussian_noise(Im_data, s=30)

bounds = {'vmin': 0, 'vmax': 255}
plt.imshow(Im_noised, cmap='gray', **bounds)

plt.show()


# #### Convergence criteria




def convergence_criteria(u0, u1, conv_crit):
    pass


# ## 3




def Denoise_Tikhonov(f, K, lamb, tau=None):
    pass


# ## 5.1
# ### Test of `Denoise_Tikhonov` 




pass


# ## 4




def Denoise_TV(f, K, lamb, eps, tau):
    pass


# ## 5.2
# ### Test of `Denoise_TV`




pass


# ## 6




def Denoise_Tikhonov_Fourier(f, lamb):
    pass


# ### Test of `Denoise_Tikhonov_Fourier` 




pass


# ## 7

# On paper

# # Extensions to deconvolution and inpainting

# ## 8

# On paper

# #### Load images




# Image : Parrot
Im_parrot = load_img('Im1.png')
Mask_parrot = load_img('Im1_mask.png')

# Image : Beach
Im_beach = load_img('Im2.png')
Mask_beach = load_img('Im2_mask.png')

# Image : Mystery text
Im_mystery = load_img('Im3.png')
Mask_mystery = load_img('Im3_mask.png')





def gaussian_kernel(size, sigma):
    variance = sigma**2
    half_size = size // 2
    
    range_matrix = np.arange(- half_size, half_size + 1)
    X, Y = np.meshgrid(range_matrix, range_matrix)
    
    return np.exp(- (X**2 + Y**2) / (2 * variance)) / (2 * np.pi * variance)





def convolve(f, G):
    return convolve2d(f, G, mode='same', boundary='symm')


# ## 9




def Deconvolution_TV(f, G, tau, eps, K, lamb):
    pass





G = None
Im_convolved = None
Im_convolved_noised = None


# ## 12.1
# ### Test of `Deconvolution_TV`




pass


# ## 10




def Inpainting_TV(f, M, tau, eps, K, lamb):
    pass


# ## 12.2
# ### Test of `Inpainting_TV`




pass


# ## 11




def Inpainting_Tikhonov(f, M, tau, K, lamb):
    pass


# ## 12.3
# ### Test of `Inpainting_Tikhonov`




pass


# ## 13

# On paper

# ## 14




def Denoise_g1(f, K, lamb, eps, tau):
    pass





def Denoise_g2(f, K, lamb, eps, tau):
    pass


# #### Comparison `Denoise_TV`, `Denoise_g1`, `Denoise_g2`




pass

