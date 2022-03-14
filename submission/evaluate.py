import numpy as np
import torch

from climatehack import BaseEvaluator
from model import Model


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        self.model = Model()
        self.model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        self.model.eval()

    def pre_process(self, data, max_value=1023):
        t, h, w = data.shape
        data = torch.from_numpy(data).view(-1,12,1,128,128)
        #print(data.shape)
        #print(t, h, w)
        #print(h//2-32,h//2+32,w//2-32,w//+32)
        data = data[:,:,:,h//2-32:h//2+32,w//2-32:w//2+32] # Cut out center 64 cells
        #print(data.shape)
        data = data/max_value
        return data

    def single_pass(self, data):
        # My model outputs 24 sequence elements.
        #print("data in pass", data.shape)
        assert data.squeeze().shape == (12,64,64)
        with torch.no_grad():
            last_state_list, layer_output = (
                self.model(data)
            )
            prediction = last_state_list[0][0][:1,:1].view(1, 64, 64).detach().numpy()
        return prediction

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """
        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        #print(data.shape)

        data = self.pre_process(data)

        # Create final 24 step prediction from single time step predictions
        prediction = []
        data_tmp = data#.copy()
        while len(prediction)<24:
            single_prediction = self.single_pass(data_tmp)
            #print("single_prediction.shape",single_prediction.shape)
            prediction.append(single_prediction.squeeze())
            data_tmp = torch.from_numpy(np.concatenate((data[:,1:], single_prediction.reshape(1,1,1,64,64)), axis=1))
        prediction = np.stack(prediction, axis=0)

        # If our model predicts all 24 steps:
        #prediction = self.single_pass(data)

        assert prediction.shape == (24, 64, 64)

        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
