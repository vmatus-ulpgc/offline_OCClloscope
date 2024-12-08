import numpy as np
import cv2
import collections
from sklearn.mixture import GaussianMixture as GM
import matplotlib.pyplot as plt
import os
plt.switch_backend('QT5Agg')

# 4-level experiments:
#cap, x,y = cv2.VideoCapture("/home/occdevkit/test4.h264"), 295, 368
#cap, fps, x,y = cv2.VideoCapture("/home/occdevkit/test6.h264"), 60, 290, 209
#cap, fps, x,y = cv2.VideoCapture("/home/occdevkit/test7.h264"), 60, 375, 190
#cap, fps, x,y = cv2.VideoCapture("/home/occdevkit/test9.h264"), 90, 327, 62
#cap, fps, x,y = cv2.VideoCapture("/home/occdevkit/2024csndsp/test10.h264"), 90, 327, 62
#cap, fps, x,y = cv2.VideoCapture("/home/occdevkit/2024csndsp/test4.h264"), 30, 352, 202

# 8-level experiments:
#cap, fps, x,y = cv2.VideoCapture("/home/occdevkit/2024csndsp/test3.h264"), 150, 491, 101
#path = "C:\\Users\\VMI_DKRM_CSI\\Desktop\\2024csndsp\\test_captures"
# path = "C:\\Users\\Y6082772\\OneDrive - ituav\\Desktop\\2024csndsp\\final_captures"
#C:\Users\Y6082772\Desktop\2025iet_oe-occpam
# path = "C:\\Users\\Y6082772\\Desktop\\2025iet_oe-occpam\\20241121_23h40"
path = "C:\\Users\\Y6082772\\Desktop\\2025iet_oe-occpam"

#ilename, fps, x,y = "test4.h264", 150, 513, 151
# filename, fps, x,y = "pam_classic1.h264", 150, 509, 174
# filename, fps, x,y = "2024-04-12-sccp-pam-k1-ag16.h264", 75, 304, 233
# filename, fps, x,y = "2024-04-19-sccppam-15m.h264", 100, 288, 247


#PAPER OPTOELECTRONICS
# filename, fps, x,y = "20241121_23h40_ag6.0_ss2000.h264", 100, 271 , 285

#5 METERS:
#filename, fps, x,y = "20241206_05m\\20241206_05m_ag1.0_ss500.h264", 100, 392 , 331

#40 METERS:
filename, fps, x,y = "20241206_40m\\20241206_40m_ag10.0_ss500.h264", 100, 322 , 290


ground_truth_file = "cppam_othman.txt"

x1, y1 = x-10, y-10
x2, y2 = x+10, y+10

x = x-x1
y = y-y1

fig_scale = 80



# # Read the file
# values = []
# transmission_identifiers = []
# with open(os.path.join(path,ground_truth_file), 'r') as file:
#     print(file.readline())
#     for line in file:
#         line = line.strip()  # Remove leading/trailing whitespaces
#         if line.startswith("Transmission identifier: "):
#             # Extract the identifier and convert it to integer
#             identifier = int(line[len("Transmission identifier: "):])
#             transmission_identifiers.append(identifier)
#         else:
#             # If not a special line, convert to integer and append to values list
#             values.append(line)

# # Convert the lists to NumPy arrays
# values_array = np.array(values)
# identifiers_array = np.array(transmission_identifiers)

# # Display the NumPy arrays
# print("\nTransmission Identifiers array:")
# print(identifiers_array)


cap = cv2.VideoCapture(os.path.join(path,filename))
pam_levels = 8
live_plot = True
gamma = 0.7
# discard_time = 3000*fps+5000
discard_time = 1

i = 0
buffer_size = int(512)
signal_buff = collections.deque(maxlen = buffer_size)
signal_buff.extend(np.zeros(buffer_size))

signal_x = np.linspace(0,buffer_size, buffer_size) #fix this
signal_y = np.array(signal_buff)

text_pxintensity = "Intensity"
text_frames = "Video frames ()"
plot_linewidth=0.8
plot_markersize=3

if live_plot:
    #
    fig = plt.figure(1,(9,3),fig_scale)
    plt.ion()
    ax = fig.add_subplot(111)
    line1, = ax.plot(signal_x,signal_y,'.-',linewidth=plot_linewidth, markersize=plot_markersize)
    #plt.ylabel("Single pixel intensity ()")
    plt.ylabel(text_pxintensity)
    plt.xlabel(text_frames)
    ax.set_ylim(0.0,255.0/255.0)
    ax.set_xlim(int(0.5*buffer_size),buffer_size)
    plt.tight_layout()
    plt.show()
 
for k in range(discard_time): #discard first n seconds of capture
    good_frame, frame = cap.read()

psf_span = 3
psf = frame[y-psf_span:y+psf_span+1,x-psf_span:x+psf_span+1,0]
# print(psf)
extent = [-psf_span, psf_span, -psf_span, psf_span]
# fig_psf = plt.figure(2,(2,2),fig_scale)
# plt.imshow(psf, cmap='gray', interpolation='nearest', extent=extent)
# plt.colorbar()
# plt.xlabel("X position [px]")
# plt.ylabel("Y position [px]")
# plt.tight_layout()
# plt.show()


if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
    cv2.destroyAllWindows()
    

while True:
    good_frame, frame = cap.read()
    if good_frame == False:
        cv2.destroyAllWindows()
        break
    
    frame = frame[y1:y2, x1:x2, :]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = np.power(frame/255.0,gamma)
    #px_val = np.mean((frame[y,x,0],frame[y,x,1],frame[y,x,2]))
    px_val = frame_gray[y,x]
    #nine_px_val = np.mean((frame[y-1,x-1,0],frame[y-1,x,0],frame[y-1,x+1,0],frame[y,x-1,0],frame[y,x,0],frame[y,x+1,0],frame[y+1,x-1,0],frame[y+1,x,0],frame[y+1,x+1,0]))
    nine_px_val = np.mean((frame_gray [y-1,x-1],frame_gray [y-1,x],frame_gray [y-1,x+1],frame_gray [y,x-1],frame_gray [y,x],frame_gray [y,x+1],frame_gray [y+1,x-1],frame_gray [y+1,x],frame_gray [y+1,x+1]))
    twelve_px_val = np.mean((frame_gray [y-2,x-2],frame_gray [y-2,x-1],frame_gray [y-1,x-2],frame_gray [y-1,x-1],frame_gray [y-1,x],frame_gray [y-1,x+1],frame_gray [y,x-1],frame_gray [y,x],frame_gray [y,x+1],frame_gray [y+1,x-1],frame_gray [y+1,x],frame_gray [y+1,x+1]))
   # four_px_val = np.mean((frame_gray[y-1,x-1],frame_gray[y-1,x],frame_gray[y,x-1],frame_gray[y,x]))
    val = nine_px_val
    # val = nine_px_val
    val = np.power(val/255.0,gamma)
    #print(val)
    signal_buff.append(val)
    signal_y = np.array(signal_buff)
    signal_max = np.max(signal_y)
    signal_min = np.min(signal_y)
    #print(i,"\t",x,"\t",y,"\t",px_val)
    if live_plot:
        line1.set_ydata(signal_y)
        fig.canvas.draw()
        fig.canvas.flush_events()
        psf = frame_gray[y-psf_span:y+psf_span+1,x-psf_span:x+psf_span+1]
    i+=1
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) 
    cv2.imshow("Camera",psf)
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        cv2.destroyAllWindows()
    if i%buffer_size == 0:
        print("buffer filled")
        plt.figure(buffer_size,(4,3.3),fig_scale)
        plt.cla()

        #ax2 = fig2.add_subplot(111)
        plt.hist(signal_y, bins=100)
        plt.ion()
        
        plt.ylabel("Counts")
        plt.xlabel("Received signal intensity ()")
        plt.xlim(signal_min,signal_max)
        plt.ylim(0,buffer_size/8)
        gmm = GM(pam_levels)
        gmm.fit(signal_y.reshape(-1, 1))
        means = gmm.means_.flatten()
        sorted_means = np.sort(means)
        thresholds = []
        for j in range(pam_levels-1):
            threshold = (sorted_means[j] + sorted_means[j+1]) / 2
            thresholds.append(threshold)
            plt.axvline(threshold, color='red', linestyle='--',linewidth=plot_linewidth)
            
            #draw them on the main oscilloscope plot
        if live_plot:
            ax.cla()
            #ax = fig.add_subplot(111)
            line1, = ax.plot(signal_x,signal_y,'.-',linewidth=plot_linewidth, markersize=plot_markersize)
            ax.set_ylabel(text_pxintensity)
            ax.set_xlabel(text_frames)
            #plt.ylabel("Single pixel intensity ()")
            #plt.ylabel(text_pxintensity)
            #plt.xlabel(text_frames)
            ax.set_ylim(0.0,255.0/255.0)
            ax.set_xlim(int(0.5*buffer_size),buffer_size)
            #plt.tight_layout()
            #plt.show()
            for j in range(pam_levels-1):
                ax.axhline(thresholds[j], color='red', linestyle='--',linewidth=plot_linewidth)
        print(thresholds)
        
        
        
        plt.tight_layout()
        plt.draw()
        plt.show()
        plt.pause(1)
        #fig2.canvas.draw()
        #fig2.canvas.flush_events()
    
        
cap.release()
