import obspy
from obspy import UTCDateTime
import numpy as np
from obspy.clients.fdsn.client import Client


class WavePreproccesing:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        data = self.x
        
        tr_E = obspy.Trace(data=data[:, 0])
        tr_E.stats.starttime = UTCDateTime(self.y[11])
        tr_E.stats.delta = 0.01
        tr_E.stats.channel = self.y[6] + "E"
        tr_E.stats.station = self.y[7]
        tr_E.stats.network = self.y[10]

        tr_N = obspy.Trace(data=data[:, 1])
        tr_N.stats.starttime = UTCDateTime(self.y[11])
        tr_N.stats.delta = 0.01
        tr_N.stats.channel = self.y[6] + "N"
        tr_N.stats.station = self.y[7]
        tr_N.stats.network = self.y[10]

        tr_Z = obspy.Trace(data=data[:, 2])
        tr_Z.stats.starttime = UTCDateTime(self.y[11])
        tr_Z.stats.delta = 0.01
        tr_Z.stats.channel = self.y[6] + "Z"
        tr_Z.stats.station = self.y[7]
        tr_Z.stats.network = self.y[10]

        self.stream = obspy.Stream([tr_E, tr_N, tr_Z])
    
    def get(self):
        for i in range(0, 3):
            tr = self.stream[i]
            tr.detrend("linear")
            tr.detrend("demean")
            tr.taper(max_percentage=0.05, type="cosine")
            tr.filter("bandpass", freqmin=0.1, freqmax=45, corners=2, zerophase=False)
        
        a = self.stream
        return np.array([a[0].data, a[1].data, a[2].data]).astype(dtype=np.float32).T

class WavePreproccesingFromHDF:
    def __init__(self, hdf):
        self.hdf = hdf
        data = np.array(hdf)[:, :3]
        
        tr_E = obspy.Trace(data=data[:, 0])
        tr_E.stats.starttime = UTCDateTime(self.hdf.attrs['trace_start_time'])
        tr_E.stats.delta = 0.01
        tr_E.stats.channel = self.hdf.attrs['receiver_type']+'E'
        tr_E.stats.station = self.hdf.attrs['receiver_code']
        tr_E.stats.network = self.hdf.attrs['network_code']

        tr_N = obspy.Trace(data=data[:, 0])
        tr_N.stats.starttime = UTCDateTime(self.hdf.attrs['trace_start_time'])
        tr_N.stats.delta = 0.01
        tr_N.stats.channel = self.hdf.attrs['receiver_type']+'N'
        tr_N.stats.station = self.hdf.attrs['receiver_code']
        tr_N.stats.network = self.hdf.attrs['network_code']

        tr_Z = obspy.Trace(data=data[:, 0])
        tr_Z.stats.starttime = UTCDateTime(self.hdf.attrs['trace_start_time'])
        tr_Z.stats.delta = 0.01
        tr_Z.stats.channel = self.hdf.attrs['receiver_type']+'Z'
        tr_Z.stats.station = self.hdf.attrs['receiver_code']
        tr_Z.stats.network = self.hdf.attrs['network_code']

        self.stream = obspy.Stream([tr_E, tr_N, tr_Z])
    
    def get(self):
        for i in range(0, 3):
            tr = self.stream[i]
            tr.detrend("linear")
            tr.detrend("demean")
            tr.taper(max_percentage=0.05, type="cosine")
            tr.filter("bandpass", freqmin=0.1, freqmax=45, corners=2, zerophase=False)
        
        a = self.stream
        return np.array([a[0].data, a[1].data, a[2].data]).astype(dtype=np.float32).T
