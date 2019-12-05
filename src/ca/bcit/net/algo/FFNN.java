package ca.bcit.net.algo;

import ca.bcit.net.*;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class FFNN implements IRMSAAlgorithm {
    MultiLayerNetwork multiLayerNetwork;
    public FFNN(){
        NN nn = new NN();
        multiLayerNetwork = nn.getNeuralNetwork();
    }
    @Override
    public String getKey() {
        return "Feed Forward Neural Network";
    }

    @Override
    public String getName() {
        return "Feed Forward Neural Network";
    }

    @Override
    public String getDocumentationURL() {
        return null;
    }

    @Override
    public DemandAllocationResult allocateDemand(Demand demand, Network network) {
        int volume = (int) Math.ceil(demand.getVolume() / 10.0) - 1;
        List<PartedPath> candidatePaths = demand.getCandidatePaths(false, network);
        sortByLength(network, volume, candidatePaths);
        if (candidatePaths.isEmpty())
            return DemandAllocationResult.NO_SPECTRUM;
        boolean workingPathSuccess = false;
        try {
            if (candidatePaths.size() == 1) {
                if (demand.allocate(network, candidatePaths.get(0)))
                    workingPathSuccess = true;
            } else {
                candidatePaths = findBestPath(volume, candidatePaths, 1);
                if (demand.allocate(network, candidatePaths.get(0)))
                    workingPathSuccess = true;
            }

        } catch (NetworkException storage) {
            return DemandAllocationResult.NO_REGENERATORS;
        }
        if (!workingPathSuccess)
            return DemandAllocationResult.NO_SPECTRUM;
        return new DemandAllocationResult(demand.getWorkingPath());
    }

    private List<PartedPath> findBestPath(int volume, List<PartedPath> candidatePaths, int start) {
        if (candidatePaths.size() <= 1)
            return candidatePaths;
        ArrayList<PartedPath> pathList1 = new ArrayList<>();
        ArrayList<PartedPath> pathList2 = new ArrayList<>();
        PartedPath path1 = candidatePaths.get(0);
        pathList1.add(path1);
        PartedPath path2 = candidatePaths.get(0);
        try {
            for (int i = 1; i < candidatePaths.size(); i++) {
                if (candidatePaths.get(i).getParts().get(start).getSource() == path1.getParts().get(start).getSource())
                    pathList1.add(candidatePaths.get(i));
                else if (candidatePaths.get(i).getParts().get(start).getSource() == path2.getParts().get(start).getSource())
                    pathList2.add(candidatePaths.get(i));
                else if (pathList2.size() == 0) {
                    path2 = candidatePaths.get(i);
                    pathList2.add(path2);
                }
            }
            if (pathList2.isEmpty())
                return findBestPath(volume, pathList1, start + 1);
            INDArray output = multiLayerNetwork.output(Nd4j.create(new double[]{
                    path1.getParts().get(0).getSource().getPosition().getX(),
                    path1.getParts().get(start).getSource().getPosition().getX(),
                    path2.getParts().get(start).getSource().getPosition().getX(),
                    path1.getParts().get(0).getSource().getPosition().getY(),
                    path1.getParts().get(start).getSource().getPosition().getY(),
                    path2.getParts().get(start).getSource().getPosition().getY(),
                    path1.getParts().get(start).getSource().getFreeRegenerators(),
                    path2.getParts().get(start).getSource().getFreeRegenerators(),
                    path1.getParts().get(start).getOccupiedSlicesPercentage(),
                    path2.getParts().get(start).getOccupiedSlicesPercentage(),
                    volume
            }));
            return (output.getDouble(0) > output.getDouble(1)) ?
                    findBestPath(volume, pathList1, start + 1) : findBestPath(volume, pathList2, start + 1);
        } catch (Exception ignored) {
            ArrayList<PartedPath> list = new ArrayList<>();
            list.add(path1);
            return list;
        }
    }

    private static List<PartedPath> sortByLength(Network network, int volume, List<PartedPath> candidatePaths) {
        pathLoop:
        for (PartedPath path : candidatePaths) {
            path.setMetric(path.getPath().getLength());
            // choosing modulations for parts
            for (PathPart part : path) {
                for (Modulation modulation : network.getAllowedModulations())
                    if (modulation.modulationDistances[volume] >= part.getLength()) {
                        part.setModulation(modulation, 1);
                        break;
                    }
                if (part.getModulation() == null)
                    continue pathLoop;
            }
        }
        for (int i = 0; i < candidatePaths.size(); i++)
            for (PathPart spec : candidatePaths.get(i).getParts())
                if (spec.getOccupiedSlicesPercentage() > 80.0) {
                    candidatePaths.remove(i);
                    i--;
                }

        candidatePaths.sort(PartedPath::compareTo);

        return candidatePaths;
    }

    public static void writeTrainingData(TrainingData data) {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(data);
        System.out.println(json);

        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            String filename = "trainingData.json";
            String workingDirectory = System.getProperty("user.dir");
            String absoluteFilePath = "";

            absoluteFilePath = workingDirectory + File.separator + filename;

            System.out.println("Final filepath : " + absoluteFilePath);

            File file = new File(absoluteFilePath);
            if (!file.exists()) {
                file.createNewFile();
                FileWriter fileWriter = new FileWriter(absoluteFilePath);
                fileWriter.write(json);
                fileWriter.close();
                System.out.println("File is created");
            } else {
                System.out.println("File already exists");
                fw = new FileWriter(file.getAbsoluteFile(), true);
                bw = new BufferedWriter(fw);

                bw.write(json);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null)
                    bw.close();
                if (fw != null)
                    fw.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }


    public static ArrayList<TrainingData> readTrainingData() {
        JSONParser jsonParser = new JSONParser();
        ArrayList<TrainingData> trainingList = new ArrayList<TrainingData>();
        try (FileReader reader = new FileReader("trainingData.json")) {
            Object obj = jsonParser.parse(reader);

            JSONArray trainingData = (JSONArray) obj;

            trainingData.forEach(emp -> trainingList.add(parsetrainingData((JSONObject) emp)));

        } catch (Exception e) {
            e.printStackTrace();
        }
        return trainingList;
    }

    private static TrainingData parsetrainingData(JSONObject trainingData) {
        //Get employee object within list
        JSONObject trainingObject = (JSONObject) trainingData.get("UnicastDemand");
        double currXCord = (double) trainingObject.get("currXCord");
        double xCord1 = (double) trainingObject.get("xCord1");
        double xCord2 = (double) trainingObject.get("xCord2");
        double currYCord = (double) trainingObject.get("currYCord");
        double yCord1 = (double) trainingObject.get("yCord1");
        double yCord2 = (double) trainingObject.get("yCord2");
        int regenerators1 = (int) trainingObject.get("regenerators1");
        int regenerators2 = (int) trainingObject.get("regenerators2");
        int occupiedSpectrum1 = (int) trainingObject.get("occupiedSpectrum1");
        int occupiedSpectrum2 = (int) trainingObject.get("occupiedSpectrum2");
        int volume = (int) trainingObject.get("volume");
        int correctPath = (int) trainingObject.get("correctPath");

        return new TrainingData(currXCord, xCord1, xCord2, currYCord, yCord1, yCord2, regenerators1, regenerators2, occupiedSpectrum1, occupiedSpectrum2, volume, correctPath);
    }
}

class NN {
    private MultiLayerNetwork neuralNetwork;

    NN() {
        neuralNetwork = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAGRAD)
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(12)
                        .nOut(10)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU) //First hidden layer
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX) //Output layer
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build());
        neuralNetwork.init();
    }

    public MultiLayerNetwork getNeuralNetwork() {
        return neuralNetwork;
    }
}

class TrainingData {
    private double currXCord;
    private double xCord1;
    private double xCord2;
    private double currYCord;
    private double yCord1;
    private double yCord2;
    private int regenerators1;
    private int regenerators2;
    private double occupiedSpectrum1;
    private double occupiedSpectrum2;
    private int volume;
    private int correctPath;

    public TrainingData(double currXCord, double xCord1, double xCord2, double currYCord, double yCord1, double yCord2, int regenerators1, int regenerators2, double occupiedSpectrum1, double occupiedSpectrum2, int volume, int correctPath) {
        this.currXCord = currXCord;
        this.xCord1 = xCord1;
        this.xCord2 = xCord2;
        this.currYCord = currYCord;
        this.yCord1 = yCord1;
        this.yCord2 = yCord2;
        this.regenerators1 = regenerators1;
        this.regenerators2 = regenerators2;
        this.occupiedSpectrum1 = occupiedSpectrum1;
        this.occupiedSpectrum2 = occupiedSpectrum2;
        this.volume = volume;
        this.correctPath = correctPath;
    }
}
