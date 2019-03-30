import matplotlib.pyplot as plt
from scipy import signal 

def find(Er,k,amuse,circ):
    #print maximum eigvalue in order to find the proper height
    if not (amuse and circ):
            print(max(Er))
    # plot eigvalues for first time in order to estimate
    # the finding parameters
    plt.plot(k,Er)
    plt.waitforbuttonpress() ; plt.close()
    # Give initial parameters in order to find local maxima
    # height parameter gives a threshold on minimum height for eigvalues
    # Must be real positive value
    h = input("Give the parameter for height: ")
    # distance parameter express the distance between local maximas
    # Must be integer value
    d = input("Give the parameter for distance: ")
    # width parameter is presented in order to separate spikes from real local maximas
    # Must be integer value
    w = input("Give the parameter for width: ")
    while (h != "OK" and d != "OK" and w != "OK"):
        peaks = []
        if not (amuse^circ) :
            # in normal πCA, we reverse eigvalue matrix in order to find minima 
            E_m = - Er + max(Er)
        else:
            E_m = Er
        #We find peaks of eigvalues and plot them 
        peaks,_ = signal.find_peaks(E_m,height = float(h),distance = int(d) , width = int(w))
        plt.plot(k,Er)
        plt.plot(peaks,Er[peaks],"o")
        plt.waitforbuttonpress() ; plt.close()
        # if we are satisfied with those values, we write OK in every parameter
        h = input("Give the parameter for height: ")
        d = input("Give the parameter for distance: ")
        w = input("Give the parameter for width: ")
    return peaks