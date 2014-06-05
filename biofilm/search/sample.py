#!/usr/bin/python

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

param_ranges = {
    'stop_on_mass': 4000,
    'stop_on_time': 20000,
    'boundary_layer': 5,
    'light_penetration': (0.1, 32),
    'tension_power': (0.1, 4),
    'distance_power': (0.1, 4),
    'initial_cell_spacing': 2,
    'media_ratio': (0.01, 1.2),
    'media_monod': (0.01, 1),
    'light_monod': (0.01, 1)
}

def run_sample(runner_id):
    from biofilm import util
    util.set_h5(util.results_h5_path('sample_{0}'.format(runner_id)))

    from biofilm.model import spec, runner, result
    import numpy as np

    def sample_spec():
        params = {}
        for param, r in param_ranges.iteritems():
            if isinstance(r, tuple):
                params[param] = np.random.uniform(*r)
            else:
                params[param] = r
        return spec.Spec(**params)

    for i in range(10000):
        s = sample_spec()
        s.save()
        result.from_model(runner.run(s)).save()
        print i,

        if i % 1000 == 0:
            mail('chase.kernan+biofilm@gmail.com', 
                 'Progress for {0}'.format(runner_id), 
                 str(i))

    mail('chase.kernan+biofilm@gmail.com', 
         '{0} done'.format(runner_id), 
         '')

def mail(to, subject, text):
    msg = MIMEMultipart()

    msg['From'] = "chase.kernan@gmail.com"
    msg['To'] = to
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    mailServer = smtplib.SMTP("smtp.gmail.com", 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login("chase.kernan@gmail.com", "Lo$4Diablos")
    mailServer.sendmail("chase.kernan@gmail.com", to, msg.as_string())
    # Should be mailServer.quit(), but that crashes...
    mailServer.close()

import sys
arg = sys.argv[1]

if arg == 'analyze':
    print 'analyzing'
    from biofilm import util
    util.set_h5(util.results_h5_path('sample'))

    import numpy as np
    from biofilm.model import spec, result, analysis
    from sklearn.linear_model import LinearRegression

    param_features = []
    for param, r in param_ranges.iteritems():
        if isinstance(r, tuple):
            param_features.append(param)

    features = np.empty((result.Result.count(), len(param_features)), float)
    values = np.empty(features.shape[0], float)
    for i, res in enumerate(result.Result.all()):
        if i % 100 == 0: print i,
        for j, param in enumerate(param_features):
            features[i, j] = getattr(res.spec, param)
        values[i] = analysis.horizontal_surface_area.get_by_result(res)
    print
    print features.shape
    regression = LinearRegression()
    regression.fit(features, values, n_jobs=-1)
    print 'Features', param_features
    print 'Coeff', regression.coef_
    print 'Intercept', regression.intercept_
    print 'Params', regression.get_params()
    print 'R2', regression.score(features, values)


else:
    run_sample(arg)