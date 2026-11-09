import torch

@torch.no_grad()
def eq_activations(model,prompt):
    # run get_activations on the prompt
    _, acts = model.get_activations(prompt) #run prompt, keep activations
    eq = acts[:, -1, :]                     #keep the "=" token activations
    layers, features = torch.nonzero(eq, as_tuple=True)
    values = eq[layers, features]
    return layers.cpu(), features.cpu(), values.float().cpu() #move to cpu from gpu 
    