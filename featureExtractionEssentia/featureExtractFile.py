def featureExtractFile(curFile):
    import sys
    import numpy
    import essentia
    from essentia.streaming import MonoLoader
    from essentia.streaming import LowLevelSpectralExtractor
    from essentia.standard import YamlOutput
    from essentia.standard import YamlInput
    from essentia.standard import PoolAggregator
    from essentia.streaming import FrameCutter
    from essentia.streaming import AutoCorrelation
    import pickle
    filename = '/home/user/Desktop/soundsDB2/classifier/featureExtractionEssentia/frameSize.npz'
    npz = numpy.load(filename)
    frameSize = int(npz['frameSize'])
    # and instantiate our algorithms
    loader = MonoLoader(filename=curFile, sampleRate=8000)
    framecutter = FrameCutter(frameSize = frameSize, hopSize = frameSize/4)
    autoCorrelator = AutoCorrelation()
    
    lowLevelExtractor = LowLevelSpectralExtractor(frameSize = frameSize, hopSize = frameSize/4,sampleRate=8000)
    
    pool = essentia.Pool()
    loader.audio >> lowLevelExtractor.signal
    lowLevelExtractor.barkbands >> (pool, curFile[:-4]+'.barkbands')
    lowLevelExtractor.barkbands_kurtosis >> (pool, curFile[:-4]+'.barkbands_kurtosis')
    lowLevelExtractor.barkbands_skewness >> (pool, curFile[:-4]+'.barkbands_skewness')
    lowLevelExtractor.barkbands_spread >> (pool, curFile[:-4]+'.barkbands_spread')
    lowLevelExtractor.hfc >> (pool, curFile[:-4]+'.hfc')
    lowLevelExtractor.mfcc >> (pool, curFile[:-4]+'.mfcc')
    lowLevelExtractor.pitch >> (pool, curFile[:-4]+'.pitch')
    lowLevelExtractor.pitch_instantaneous_confidence >> (pool, curFile[:-4]+'.pitch_instantaneous_confidence')
    lowLevelExtractor.pitch_salience >> (pool, curFile[:-4]+'.pitch_salience')
    lowLevelExtractor.silence_rate_20dB >> (pool, curFile[:-4]+'.silence_rate_20dB')
    lowLevelExtractor.silence_rate_30dB  >> (pool, curFile[:-4]+'.silence_rate_30dB ')
    lowLevelExtractor.silence_rate_60dB >> (pool, curFile[:-4]+'.silence_rate_60dB')
    lowLevelExtractor.spectral_complexity >> (pool, curFile[:-4]+'.spectral_complexity')
    lowLevelExtractor.spectral_crest >> (pool, curFile[:-4]+'.spectral_crest')
    lowLevelExtractor.spectral_decrease >> (pool, curFile[:-4]+'.spectral_decrease')
    lowLevelExtractor.spectral_energy >> (pool, curFile[:-4]+'.spectral_energy')
    lowLevelExtractor.spectral_energyband_low >> (pool, curFile[:-4]+'.spectral_energyband_low')
    lowLevelExtractor.spectral_energyband_middle_low >> (pool, curFile[:-4]+'.spectral_energyband_middle_low')
    lowLevelExtractor.spectral_energyband_middle_high >> (pool, curFile[:-4]+'.spectral_energyband_middle_high')
    lowLevelExtractor.spectral_energyband_high  >> None
    lowLevelExtractor.spectral_flatness_db >> (pool, curFile[:-4]+'.spectral_flatness_db')
    lowLevelExtractor.spectral_flux >> (pool, curFile[:-4]+'.spectral_flux')
    lowLevelExtractor.spectral_rms >> (pool, curFile[:-4]+'.spectral_rms')
    lowLevelExtractor.spectral_rolloff >> (pool, curFile[:-4]+'.spectral_rolloff')
    lowLevelExtractor.spectral_strongpeak >> (pool, curFile[:-4]+'.spectral_strongpeak')
    lowLevelExtractor.zerocrossingrate >> (pool, curFile[:-4]+'.zerocrossingrate')
    lowLevelExtractor.inharmonicity >> (pool, curFile[:-4]+'.inharmonicity')
    lowLevelExtractor.tristimulus >> (pool, curFile[:-4]+'.tristimulus')
    lowLevelExtractor.oddtoevenharmonicenergyratio >> (pool, curFile[:-4]+'.oddtoevenharmonicenergyratio')
    lowLevelExtractor.inharmonicity >> None
    lowLevelExtractor.tristimulus >> None
    lowLevelExtractor.oddtoevenharmonicenergyratio >> None



    #mfcc.bands >> (pool, curFile[:-4]+'.mfccBands')
    #mfcc.mfcc >> (pool, curFile[:-4]+'.mfcc')

    essentia.run(loader)
    aggrPool = PoolAggregator(defaultStats = ['min', 'max', 'median', 'mean', 'var', 'skew', 'kurt', 'dmean', 'dvar'])(pool)
    #aggrPool = PoolAggregator(defaultStats = ['min', 'max', 'mean', 'var'])(pool)
    YamlOutput(filename=curFile[:-4]+'trainingFeatures.yaml', format="yaml")(aggrPool)
    essentia.reset(loader)
    return




