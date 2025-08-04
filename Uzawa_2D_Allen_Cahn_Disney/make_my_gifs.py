import imageio.v2 as imageio
from pdf2image import convert_from_path
state = []
control = []
adjnt = []
for epoch in range(0,2500,50):
    statepng = convert_from_path(f'./Vary_alpha_8_nPts_159_eps_0.01x5/State_frame_{epoch}.pdf')
    statepng[0].save(f'./Vary_alpha_8_nPts_159_eps_0.01x5/State_frame_{epoch}.png', 'PNG')
    state.append(imageio.imread(f'./Vary_alpha_8_nPts_159_eps_0.01x5/State_frame_{epoch}.png'))

    #contrpng = convert_from_path(f'./Vary_alpha_6_nPts_159_eps_0.01x1/Cntrl_frame_{epoch}.pdf')
    #contrpng[0].save(f'./Vary_alpha_6_nPts_159_eps_0.01x1/Cntrl_frame_{epoch}.png', 'PNG')
    #control.append(imageio.imread(f'./Vary_alpha_6_nPts_159_eps_0.01x1/Cntrl_frame_{epoch}.png'))
    
imageio.mimsave('./alpha_8_nPts_159_eps_0.01x5.gif', state)
#imageio.mimsave('./alpha_6_nPts_159_eps_0.1x1_control.gif', control)
