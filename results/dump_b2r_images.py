
import cv2
import numpy as np
import os
from biofilm import util

def dump(h5, dump_dir, spec_query=None, naming_fields=None):
    util.set_h5(util.results_h5_path(h5))
    
    try:
        os.makedirs(dump_dir)
    except:
        pass

    from biofilm.model.result import Result
    from biofilm.model.spec import Spec
    from biofilm.model import analysis as an

    print spec_query
    if spec_query:
        specs = Spec.where(spec_query)
    else:
        specs = Spec.all()

    for spec in specs:
        results = Result.where('spec_uuid=="{0}"'.format(spec.uuid))
        for i, result in enumerate(results):
            if naming_fields:
                name = "-".join("{0}{1:.3f}".format(f, getattr(spec, f))
                                for f in naming_fields.split(','))
                name += "-density{:.3f}".format(an.convex_density.func(result))
                name += "-{0}.png".format(i)
            else:
                name = "{0}-{1}.png".format(spec.uuid, result.uuid)
            path = os.path.join(dump_dir, name)
            
            image = cv2.resize(result.int_image[::-1, :]*255, None, None, 
                               4, 4, cv2.INTER_NEAREST)
            color = np.zeros(image.shape + (3,), dtype=np.uint8) 
            color[:, :, 2] = image
            cv2.imwrite(path, color)

if __name__ == '__main__':
    import sys
    dump('b2r', 'model_images/b2r/low', '((boundary_layer**2)*(media_ratio))<4',
         naming_fields='boundary_layer,media_ratio')
    # dump('b2r', 'model_images/b2r/high', '(26<((boundary_layer**2)*(media_ratio)))&(((boundary_layer**2)*(media_ratio))<28)',
    #     naming_fields='boundary_layer,media_ratio')

