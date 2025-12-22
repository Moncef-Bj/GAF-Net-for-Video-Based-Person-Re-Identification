import numpy as np
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    Args:
     data_list_path (string):   Path to pose data.
     sequence_length:           Length of sequence for each data point. The number of frames of pose data returned.
     train:                     Training dataset or validation. default : True
     transform:                 Transformation on the dataset
     target_transform:          Transformation on the target.
    """

    def __init__(
        self,
        data_list_path,
        sequence_length=1,
        train=True,
        transform=None,
        target_transform=None,
    ):
        super(PoseDataset, self).__init__()
        self.data_list = np.loadtxt(data_list_path, skiprows=1, dtype=str)
        self.sequence_length = sequence_length
        self.train = train

        self.transform = transform
        self.target_transform = target_transform

        self.data_dict = {}

        for row in self.data_list:
            row = row.split(",")
            if row[0][:4]=='00-1':
                continue 
            target, frame_num = self._filename_to_target(row[0])

            if target not in self.data_dict:
                self.data_dict[target] = {}

            if len(row[1:]) != 51:
                print("Invalid pose data for: ", target, ", frame: ", frame_num)
                continue
            # Added try block to see if all the joint values are present. other wise skip that frame.
            try:
                self.data_dict[target][frame_num] = np.array(
                    row[1:], dtype=np.float32
                ).reshape((-1, 3))
            except ValueError:
                print("Invalid pose data for: ", target, ", frame: ", frame_num)
                continue

        # Check for data samples that have less than sequence_length frames and remove them.
        for target, sequence in self.data_dict.copy().items():
            l=list(sequence.keys())[-1]
            l1=list(sequence.keys())

            if len(sequence) < self.sequence_length + 1:
                # dif=self.sequence_length + 1-len(sequence)
                # for j in range(dif):
                #      sequence[l+j+1]=sequence[l1[j%len(l1)]]

                del self.data_dict[target]

        self.targets = list(self.data_dict.keys())

        self.data = list(self.data_dict.values())

    def _filename_to_target(self, filename):
        raise NotImplemented()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (pose, target) where target is index of the target class.
        """
        target = self.targets[index]
        data = np.stack(list(self.data[index].values()))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def get_num_classes(self):
        """
        Returns number of unique ids present in the dataset. Useful for classification networks.

        """
        if type(self.targets[0]) == int:
            classes = set(self.targets)
        else:
            classes = set([target[0] for target in self.targets])
        num_classes = len(classes)
        return num_classes


class CasiaBPose(PoseDataset):
    """
    CASIA-B Dataset
    The format of the video filename in Dataset B is 'xxx-mm-nn-ttt.avi', where
      xxx: subject id, from 001 to 124.
      mm: walking status, can be 'nm' (normal), 'cl' (in a coat) or 'bg' (with a bag).
      nn: sequence number.
      ttt: view angle, can be '000', '018', ..., '180'.
     """

    mapping_walking_status = {
        'nm': 0,
        'bg': 1,
        'cl': 2,
    }

    def _filename_to_target(self, filename):
        _, sequence_id, frame = filename.split("/")
        subject_id, walking_status, sequence_num, view_angle = sequence_id.split("-")
        walking_status = self.mapping_walking_status[walking_status]
        return (
            (int(subject_id), int(walking_status), int(sequence_num), int(view_angle)),
            int(frame[:-4]),
        )


class KinectGait(PoseDataset):
    def _filename_to_target(self, filename):
        subject_id, sequence_num, frame = filename.split("-")
        return (int(subject_id), int(sequence_num)), int(frame)


class iLIDS(PoseDataset):
    """ 
    ILIDs Dataset:
    The Format of the video filename is ILIDSis n bbox "cam2_person043_01561.png",
   
    """
    camera={
        'cam1':1,
        'cam2':2,
        
    }
    def _filename_to_target(self, filename):
        subject_id=filename[11:14]
        id_camera=filename[:4]
        frame=filename[15:20]
        id_camera=self.camera[id_camera]
        
        return ((int(subject_id),int(id_camera)),int(frame))
    
class MARS(PoseDataset):
    """ 
    MARS Dataset:
    The Format of the video filename is MARS is n bbox "0065C1T0002F0016.jpg",
    "0065" is the ID of the pedestrian. "C1" denotes the first camera (there are totally 6 cameras). "T0002" means the 2th tracklet. "F016" is the 16th frame within this tracklet. 
    For the tracklets, their names are accumulated for each ID; but for frames, they start from "F001" in each tracklet.
    """
    camera={
        'C1':1,
        'C2':2,
        'C3':3,
        'C4':4,
        'C5':5,
        'C6':6,
        
    }
    def _filename_to_target(self, filename):
        subject_id=filename[:4]
        id_camera=filename[4:6]
        sequence_num=filename[7:11]
        frame=filename[12:15]
        id_camera=self.camera[id_camera]

        return(
            (int(subject_id),int(id_camera),int(sequence_num)),int(frame)
            )