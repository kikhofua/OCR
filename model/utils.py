from collections import namedtuple

VFMDims = namedtuple('VisualFeatureMapDimensions', 'D_tick H W')
MLPSpecs = namedtuple('MultilayeredPerceptronSpecs', 'hidden_sizes output_size hidden_act_fn output_act_fn')