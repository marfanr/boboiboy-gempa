import pandas as pd
import h5py
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

class DataSplitter():
    def __init__(self, hdf5, csv, out):
        self.csv = csv
        self.hdf5_path = hdf5
        self.out = out
        
        df = pd.read_csv(csv)
        df = df[df.trace_category == 'earthquake_local']
        event_ids = df["source_id"].unique()
        train_events, test_events = train_test_split(
            event_ids,
            test_size=0.2,
            shuffle=False,
        )
        df_train = df[df.source_id.isin(train_events)]
        df_test  = df[df.source_id.isin(test_events)]


        self.ev_list = df_train['trace_name'].to_list()
        self.ev_list_test = df_test['trace_name'].to_list()

        print(len(self.ev_list), len(self.ev_list_test))
        
        MAX_GB = 8.0
        self.MAX_BYTES = MAX_GB * 1024 * 1024 * 1024

        self.x_data, self.self.y_main, self.y_meta = [], [], []
        self.current_bytes = 0
        self.current_bytes
    
    def flush_to_disk(self, name):
        if len(self.x_data) == 0:
            return

        x_arr = np.stack(self.x_data, dtype=np.float32)
        y_arr = np.stack(self.y_main, dtype=np.float32)
        meta_arr = np.array(self.y_meta, dtype=object)

        filename = f"{self.out}/stead_chunk_{name}_{file_index}.npz"
        np.savez_compressed(filename, x=x_arr, y=y_arr, meta=meta_arr)

        print(f"[SAVE] {filename} | x={x_arr.shape} y={y_arr.shape} meta={meta_arr.shape}")

        self.x_data.clear()
        self.y_main.clear()
        self.y_meta.clear()
        self.current_bytes = 0
        file_index += 1
        
        
    def worker_init(self, hdf5_path):
        global HDF5_FILE
        HDF5_FILE = h5py.File(hdf5_path, 'r', libver='latest', swmr=True)

    def process_event(self, evi):
        """
        Jalan dalam setiap process. Tidak meng-open file lagi.
        """
        dataset = HDF5_FILE['data/' + str(evi)]
        data = np.array(dataset)[:, :3].astype(np.float32)  # float32 lebih kecil

        attrs = dataset.attrs
        y = np.array([
            attrs.get('p_arrival_sample', 0),
            attrs.get('s_arrival_sample', 0),
            attrs.get('p_travel_sec', 0),
            1,  # sudah pasti earthquake_local
            attrs.get('source_latitude', 0),
            attrs.get('source_longitude', 0),
            # string tidak bisa dicampur, keluarkan ke structured array
        ], dtype=np.float32)

        # sisakan string metadata ke tuple
        meta = (
            attrs.get('receiver_type', ''),
            attrs.get('receiver_code', ''),
            attrs.get('receiver_latitude', 0),
            attrs.get('receiver_longitude', 0),
            attrs.get('network_code', ''),
            attrs.get('trace_start_time', '')
        )

        return data, y, meta
    

    def split(self, batch=2000):
        BATCH = 2000  # submit 2000 task per batch, jauh lebih efisien
        # 118000
        with ProcessPoolExecutor(
            max_workers=12,
            initializer=self.worker_init,
            initargs=(self.hdf5_path,)
        ) as executor:

            for start in range(118000, len(self.ev_list), BATCH):
                batch = self.ev_list[start:start+BATCH]

                futures = {executor.submit(self.process_event, e): e for e in batch}

                for i, fut in enumerate(as_completed(futures)):
                    data, y, meta = fut.result()

                    self.x_data.append(data)
                    self.y_main.append(y)
                    self.y_meta.append(meta)

                    current_bytes += data.nbytes + y.nbytes + sys.getsizeof(meta)

                    if current_bytes >= self.MAX_BYTES:
                        self.flush_to_disk("train")

                print(f"[BATCH OK] {start}/{len(self.ev_list)} RAM={current_bytes/1e9:.3f} GB")

        self.flush_to_disk("train")
        print("DONE")
        
        self.x_data.clear()
        self.y_main.clear()
        self.y_meta.clear()

        with ProcessPoolExecutor(
            max_workers=8,
            initializer=self.worker_init,
            initargs=(self.hdf5_path,)
        ) as executor:

            for start in range(0, len(self.ev_list_test), BATCH):
                batch = self.ev_list_test[start:start+BATCH]

                futures = {executor.submit(self.process_event, e): e for e in batch}

                for i, fut in enumerate(as_completed(futures)):
                    data, y, meta = fut.result()

                    self.x_data.append(data)
                    self.y_main.append(y)
                    self.y_meta.append(meta)

                    current_bytes += data.nbytes + y.nbytes + sys.getsizeof(meta)

                    if current_bytes >= self.MAX_BYTES:
                        self.flush_to_disk("test")

                print(f"[BATCH OK] {start}/{len(self.ev_list_test)} RAM={current_bytes/1e9:.3f} GB")

        self.flush_to_disk("test")
        print("DONE")