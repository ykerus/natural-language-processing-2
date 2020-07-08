import torch
import numpy as np

def generate_data(shape, size=28, num_samples=500):

    data = torch.zeros((num_samples, 1, size, size))
    attr = torch.zeros((num_samples, 3)) 
    targets = torch.zeros((num_samples, 3)) 

    if shape == "circle":
        for n in range(num_samples):  
            while True:
                p = torch.rand(2)-.5
                S = torch.rand(1)/2 
                c = size * (.5 + p) 
                R = size * torch.sqrt(S / np.pi) 
                if not (sum(c - R < 0) or sum(c + R >= size) or R < 1):
                    attr[n,:2], attr[n,2] = c, R
                    targets[n,:2], targets[n,2] = p, S
                    break 
        for i in range(size):
            for j in range(size):
                data[:,:,i,j][(i-attr[:,0])**2 + (j-attr[:,1])**2 <= attr[:,2]**2] = 1

    elif shape == "square":
        for n in range(num_samples):  
            while True:
                p = torch.rand(2)-.5
                S = torch.rand(1)/2

                c = size * (.5 + p) 
                Z = size * torch.sqrt(S)/2 # center to edge distance

                if not (sum(c - Z < 0) or sum(c + Z >= size) or Z < .5):
                    attr[n,:2], attr[n,2] = c, Z
                    targets[n,:2], targets[n,2] = p, S
                    break # successful
        for i in range(size):
            for j in range(size):
                mask = (i >= attr[:,0]-attr[:,2]) & (i <= attr[:,0]+attr[:,2]) &\
                       (j >= attr[:,1]-attr[:,2]) & (j <= attr[:,1]+attr[:,2])
                data[:,:,i,j][mask] = 1
    return data, targets
    
def reconstruct(shape, attributes, size=28):
    bsize = attributes.size(0)
    
    data = torch.zeros((bsize, 1, size, size))
    attr = torch.zeros((bsize,3))

    attr[:,:2] = size * (.5 + attributes[:,:2])

    if shape == "circle":
        attr[:,2] = size * torch.sqrt(attributes[:,2] / np.pi)
        for i in range(size):
            for j in range(size):
                data[:,:,i,j][(i-attr[:,0])**2 + (j-attr[:,1])**2 <= attr[:,2]**2] = 1

    elif shape == "square":
        attr[:,2] = size * torch.sqrt(attributes[:,2]) / 2
        for i in range(size):
            for j in range(size):
                mask = (i >= attr[:,0]-attr[:,2]) & (i <= attr[:,0]+attr[:,2]) &\
                       (j >= attr[:,1]-attr[:,2]) & (j <= attr[:,1]+attr[:,2])
                data[:,:,i,j][mask] = 1
    return data
    
def alter_attributes(shape, atts):
    altered = atts.clone().detach()
    if shape == "circle":
        for i in range(atts.size(0)):
            attr = torch.randint(3, (1,))
            alter = altered[i]
            while True:
                change = torch.abs(torch.rand(1)-.5) if attr == 2 else torch.rand(1)-.5
                alter[attr] = change
                R = torch.sqrt(alter[2]/np.pi)
                if not (sum(alter[:2] - R < -.5) or sum(alter[:2] + R >= .5) or R*28<1):
                    altered[i,:] = alter
                    break
    elif shape == "square":
        for i in range(atts.size(0)):
            attr = torch.randint(3, (1,))
            alter = altered[i]
            while True:
                change = torch.abs(torch.rand(1)-.5) if attr == 2 else torch.rand(1)-.5
                alter[attr] = change
                Z = 0.5 * torch.sqrt(alter[2])
                if not (sum(alter[:2] - Z < -.5) or sum(alter[:2] + Z >= .5) or Z*28<.5):
                    altered[i,:] = alter
                    break
    return altered

def get_sampledistractors(N, shape, target_atts, ref_atts, size=28):
  if shape != "square":
    raise Exception("TO DO")

  rand = torch.rand(6)
  S = torch.sqrt(ref_atts[2])/2

  max0 = 0.5 - S
  max1 = 0.5 - S 
  max2 = (1 - max(abs(ref_atts[0]),abs(ref_atts[1])))**2

  min0 = -0.5 + S
  min1 = -0.5 + S
  min2 = 1 / size**2

  high0 = rand[0]*(max0 - ref_atts[0]) + ref_atts[0]
  low0 = rand[1]*(ref_atts[0] - min0) + min0

  high1 = rand[2]*(max1 - ref_atts[1]) + ref_atts[1]
  low1 = rand[3]*(ref_atts[1] - min1) + min1

  high2 = rand[4]*(max2 - ref_atts[2]) + ref_atts[2]
  low2 = rand[5]*(ref_atts[2] - min2) + min2

  distract_atts = ref_atts.clone().repeat(N, 1)

  ref_idx = (target_atts != ref_atts).nonzero()
  ref_high = ref_atts[ref_idx] > target_atts[ref_idx]
  if ref_idx == 0:
    if ref_high:
      distract_atts[0,0] = high0
    else:
      distract_atts[0,0] = low0
    distract_atts[1,1] = high1
    distract_atts[2,1] = low1
    distract_atts[3,2] = high2
    distract_atts[4,2] = low2
  if ref_idx == 1:
    distract_atts[0,0] = high0
    distract_atts[1,0] = low0
    if ref_high:
      distract_atts[1,1] = high1
    else:
      distract_atts[1,1] = low1
    distract_atts[3,2] = high2
    distract_atts[4,2] = low2
  if ref_idx == 2:
    distract_atts[0,0] = high0
    distract_atts[1,0] = low0
    distract_atts[2,1] = high1
    distract_atts[3,1] = low1
    if ref_high:   
      distract_atts[4,2] = high2
    else:
      distract_atts[4,2] = low2
  return reconstruct(shape, distract_atts)
  
def get_distractors(N, shape, target_atts, ref_atts, size=28):
  distractors = torch.zeros((target_atts.size(0),N,1,size,size))
  for i in range(target_atts.size(0)):
    distractors[i,:,:,:,:] = get_sampledistractors(N, shape, target_atts[i], ref_atts[i], size=size)
  return distractors
  