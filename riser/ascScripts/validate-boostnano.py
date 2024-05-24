from ont_fast5_api.fast5_interface import get_fast5_file 
import ont_fast5_api as ont
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools
import h5py
import pysam
import os


import h5py
import pysam
import matplotlib.pyplot as plt
import numpy as np

fast5_filepath1 = "data/subset/group0Train/batch0.fast5" # This is IVT 2 and 4.
fast5_filepath2 = "data/subset/group1Train/batch0.fast5" #IVT 1 and 3

searchEle = ('82463218-d939-471b-902e-08f3c1fc5dce', 'IVT-i-1')
bam = "SM47_IVT_NM_RNA004.IVT-i.filtered.bam"

def bamGetIVT(readID, alignmentFile):
    
    """Returns the IVT reference from the supplied read ID.
    
    args:
    readID - The read ID to retrieve the IVT reference from
    alignmentFile - The alignment (BAM/SAM) file to use
    returns:
    reference_name: 
    """

    with pysam.AlignmentFile(alignmentFile, "rb") as alignmentFile:
        for line in alignmentFile:
            if readID == line.query_name:
                return line.reference_name

def makeFigures(filepath):
    """Saves the first 20 reads in a fast5 file as matplotlib charts
    containing the signal content before and after BoostNano trimming.
    
    args:
    filepath - Path to fast5 file
    """
    with h5py.File(f"{filepath}") as file:
        for key in list(file.keys())[:20]:
            ivt = bamGetIVT(key[5:], bam)
            sig = np.array(file[f"/{key}/Raw/Signal"])
            sigold = np.array(file[f"/{key}/Raw/Signal_Old"])

            fig, (ax1, ax2) = plt.subplots(2)
            plt.suptitle(f"{key}")

            x1 = np.arange(len(sigold))
            x2 = np.arange(len(sigold)-len(sig), len(sigold))
            ax1.plot(x1, sigold)
            ax2.set_xlim(ax1.get_xlim())
            ax2.plot(x2, sig)
            plt.savefig(f"./BoostNanoCharts/{ivt}_{key}.png")

makeFigures(fast5_filepath1)
makeFigures(fast5_filepath2)

'''
{'barcoding_enabled': '0', 'experiment_type': 'rna', 'local_basecalling': '0', 'package': 'bream4', 'package_version': '7.8.2', 'sample_frequency': '4000', 'selected_speed_bases_per_second': '130', 'sequencing_kit': 'sqk-rna004'}
{'barcoding_enabled': '0', 'experiment_type': 'rna', 'local_basecalling': '0', 'package': 'bream4', 'package_version': '7.8.2', 'sample_frequency': '4000', 'selected_speed_bases_per_second': '130', 'sequencing_kit': 'sqk-rna004'}
{'barcoding_enabled': '0', 'experiment_type': 'rna', 'local_basecalling': '0', 'package': 'bream4', 'package_version': '7.8.2', 'sample_frequency': '4000', 'selected_speed_bases_per_second': '130', 'sequencing_kit': 'sqk-rna004'}
{'barcoding_enabled': '0', 'experiment_type': 'rna', 'local_basecalling': '0', 'package': 'bream4', 'package_version': '7.8.2', 'sample_frequency': '4000', 'selected_speed_bases_per_second': '130', 'sequencing_kit': 'sqk-rna004'}
{'barcoding_enabled': '0', 'experiment_type': 'rna', 'local_basecalling': '0', 'package': 'bream4', 'package_version': '7.8.2', 'sample_frequency': '4000', 'selected_speed_bases_per_second': '130', 'sequencing_kit': 'sqk-rna004'}
(.venv) [ji0740@gadi-login-05 fast5]$ h5dump -a "/read_5b369862-2666-4869-a09e-5cdde571417b/channel_id/sampling_rate" data/subset/group0Train/batch0.fast5
HDF5 "data/subset/group0Train/batch0.fast5" {
ATTRIBUTE "sampling_rate" {
   DATATYPE  H5T_IEEE_F64LE
   DATASPACE  SCALAR
   DATA {
   (0): 4000
   }
}
}
(.venv) [ji0740@gadi-login-05 fast5]$ 
'''
