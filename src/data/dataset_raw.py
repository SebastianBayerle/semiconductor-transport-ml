from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import os  
import hashlib

class TransportDataset(Dataset):
    """
    Clean raw dataset for gather_mega_imp.hdf5.

    Loads once from HDF5 and caches:
        curves:      (N, 4, 100)  -> [L11, L12, L11B, L12B]
        parameters:  (N, 7)       -> [t, alpha, delta, gamma, eposs, eweight, accdon]
        tempaxis:    (100,)

    No preprocessing is applied here.
    """

    curve_names = ["L11", "L12", "L11B", "L12B"] 
    param_names = ["t", "alpha", "delta", "gamma", "eposs", "eweight", "accdon"]

    def __init__(self, dataset_path, cache_dir="cache", use_cache=True):
        self.dataset_path = os.path.abspath(dataset_path)
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        cache_name = self._make_cache_name()
        self.cache_path = os.path.join(self.cache_dir, cache_name)

        if use_cache and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._load_hdf5()
            self._save_cache()

    def _make_cache_name(self):
        base_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        path_hash = hashlib.md5(self.dataset_path.encode("utf-8")).hexdigest()[:8]
        return f"{base_name}_raw_cache_{path_hash}.pt"

    def _load_cache(self):
        cached = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.curves = cached["curves"]
        self.parameters = cached["parameters"]
        self.tempaxis = cached["tempaxis"]
        print(f"Loaded raw dataset from cache: {self.cache_path}")

    def _save_cache(self):
        torch.save(
            {
                "curves": self.curves,
                "parameters": self.parameters,
                "tempaxis": self.tempaxis,
            },
            self.cache_path,
        )
        print(f"Saved raw dataset to cache: {self.cache_path}")

    def _load_hdf5(self):
        curves_list = []
        params_list = []

        with h5py.File(self.dataset_path, "r") as f:
            self.tempaxis = f[".tempaxis"][()]

            for t_key in f.keys():
                if t_key == ".tempaxis":
                    continue

                grp_t = f[t_key]

                for alpha_key, grp_alpha in grp_t.items():
                    for delta_key, grp_delta in grp_alpha.items():
                        for gamma_key, grp_gamma in grp_delta.items():
                            for eposs_key, grp_eposs in grp_gamma.items():
                                for eweight_key, grp_eweight in grp_eposs.items():
                                    for accdon_key, grp_accdon in grp_eweight.items():
                                        if "kubo" not in grp_accdon:
                                            continue

                                        kubo_grp = grp_accdon["kubo"]

                                        # require all curves to exist
                                        if not all(name in kubo_grp for name in self.curve_names):
                                            continue

                                        curves = np.stack(
                                            [kubo_grp[name][()] for name in self.curve_names],
                                            axis=0,
                                        ).astype(np.float32)

                                        params = np.array(
                                            [
                                                float(t_key),
                                                float(alpha_key),
                                                float(delta_key),
                                                float(gamma_key),
                                                float(eposs_key),
                                                float(eweight_key),
                                                float(accdon_key),
                                            ],
                                            dtype=np.float32,
                                        )

                                        curves_list.append(curves)
                                        params_list.append(params)

        if not curves_list:
            raise RuntimeError("No data points loaded. Check HDF5 structure.")

        self.curves = torch.tensor(np.stack(curves_list, axis=0), dtype=torch.float32)
        self.parameters = torch.tensor(np.stack(params_list, axis=0), dtype=torch.float32)

        print("Loaded raw dataset from HDF5")
        print("curves:", self.curves.shape)
        print("parameters:", self.parameters.shape)

    def __len__(self):
        return self.curves.shape[0]

    def __getitem__(self, idx):
        return self.curves[idx], self.parameters[idx]

    def shape(self):
        return self.curves.shape, self.parameters.shape

    def get_curves(self):
        return self.curves

    def get_parameters(self):
        return self.parameters

    def get_tempaxis(self):
        return self.tempaxis