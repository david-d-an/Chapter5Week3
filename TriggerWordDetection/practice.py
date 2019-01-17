from td_utils import *


def practicefunc():

    plt.figure()
    x = graph_spectrogram("audio_examples/example_train.wav")
    plt.show(block=False)

    _, data = wavfile.read("audio_examples/example_train.wav")
    print("Time steps in audio recording before spectrogram", data[:,0].shape)
    print("Time steps in input after spectrogram", x.shape)

    activates, negatives, backgrounds = load_raw_audio()
    # Load audio segments using pydub     activates, negatives, backgrounds = load_raw_audio()
    print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
    print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
    print("activate[1] len: " + str(len(activates[1])))     # Different "activate" clips can have different lengths 

    overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
    overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
    print("Overlap 1 = ", overlap1)
    print("Overlap 2 = ", overlap2)

    np.random.seed(5)
    audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
    audio_clip.export("insert_test.wav", format="wav")
    print("Segment Time: ", segment_time)
    # IPython.display.Audio("insert_test.wav")

    # Insert audio clip into a background
    np.random.seed(5)
    audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
    audio_clip.export("insert_test.wav", format="wav")
    print("Segment Time: ", segment_time)
    # IPython.display.Audio("insert_test.wav")

    x, y = create_training_example(backgrounds[0], activates, negatives)
    plt.figure()
    plt.plot(y[0,:])
    # plt.plot(y[0])
    plt.show(block=False)
