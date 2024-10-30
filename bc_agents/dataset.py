import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import swifter
from sklearn.model_selection import train_test_split


class BCDataset(Dataset):
    def __init__(self, data_path, split='train'):
        """A dataset for behavior cloning with sequential data. 
        """
        super().__init__()
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        # Update self.data to only include rows where 'success' is 1
        self.data = self.data[self.data['success'] == 1]

        self.train_df, self.val_df = train_test_split(self.data, test_size=0.2)
        
        # split the data into train episodes
        if split == 'train':
            self.data = self.train_df
        else:
            self.data = self.val_df
        
        self.data['student_state'] = self.data['student_state'].swifter.apply(lambda x: np.fromstring(x.replace('\n','').
                                                                                              replace('[','').
                                                                                              replace(']','').
                                                                                              replace('  ',' '), 
                                                                                              sep=' ',
                                                                                              dtype=np.float32))
        # self.data['student_action'] = self.data['student_action'].swifter.apply(lambda x: np.fromstring(x.replace('\n','').
        #                                                                                       replace('[','').
        #                                                                                       replace(']','').
        #                                                                                       replace('  ',' '), 
        #                                                                                       sep=' ',
        #                                                                                       dtype=np.float32))
        self.data['expert_action'] = self.data['expert_action'].swifter.apply(lambda x: np.fromstring(x.replace('\n','').
                                                                                              replace('[','').
                                                                                              replace(']','').
                                                                                              replace('  ',' '), 
                                                                                              sep=' ',
                                                                                              dtype=np.float32))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        student_state = torch.from_numpy(self.data['student_state'].iloc[index])
        teacher_action = torch.from_numpy(self.data['expert_action'].iloc[index])
        image = Image.open(self.data['image_name'].iloc[index])
        image = np.array(image)
        observation = torch.from_numpy(image)
        return {'student_state': student_state, 'expert_action': teacher_action, 'observation': observation}