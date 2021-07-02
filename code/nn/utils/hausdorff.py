from interpret_segmentation import hdm
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def explainHausdorf(model, dataset, segment, image, save_image = False):

    # initialize the explainer with image width and height
    explainer = hdm.HausdorffDistanceMasks(240, 240)

    # generate masks
    explainer.generate_masks(circle_size=25, offset=5)

    # apply masks and calculate distances
    result = explainer.explain(model, image, segment, device)

    # generate circle map visualizations
    raw = result.circle_map(hdm.RAW, color_map='Blues')
    better = result.circle_map(hdm.BETTER_ONLY, color_map='Greens')
    worse = result.circle_map(hdm.WORSE_ONLY, color_map='Reds')

    # show with matplotlib...
    plt.imshow(raw)
    plt.show()
    
    if save_image == True
        # ...or save to disk
        raw.save('raw.png')