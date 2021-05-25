package ca.bcit.net.algo;

import ca.bcit.net.*;
import ca.bcit.net.demand.Demand;
import ca.bcit.net.demand.DemandAllocationResult;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class FFNN implements IRMSAAlgorithm {
    MultiLayerNetwork multiLayerNetwork;

    public FFNN() {
        NN nn = new NN();
        multiLayerNetwork = nn.getNeuralNetwork();
        try {
            File file = new File("trainingData.json");
            FileReader fileReader = new FileReader(file);
            JSONParser jsonParser = new JSONParser();
            JSONArray oldArray = (JSONArray) jsonParser.parse(fileReader);
            for (Object object : oldArray) {
                JSONObject jsonObject = (JSONObject) object;
                INDArray inputs = Nd4j.create(new double[]{
                        (double) jsonObject.get("currXCord"),
                        (double) jsonObject.get("xCord1"),
                        (double) jsonObject.get("xCord2"),
                        (double) jsonObject.get("currYCord"),
                        (double) jsonObject.get("yCord1"),
                        (double) jsonObject.get("yCord2"),
                        (long) jsonObject.get("regenerators1"),
                        (long) jsonObject.get("regenerators2"),
                        (double) jsonObject.get("occupiedSpectrum1"),
                        (double) jsonObject.get("occupiedSpectrum2"),
                        (long) jsonObject.get("volume"),
                });
                INDArray output = ((long) jsonObject.get("correctPath") == 0) ? Nd4j.create(new double[]{1,0}):Nd4j.create(new double[]{0,1});
                multiLayerNetwork.fit(inputs, output);
            }
            System.out.println("Training Finished");
        } catch (Exception e) {
            System.out.println("File cannot be read or doesnt exist");
            e.printStackTrace();
        }
    }

    @Override
    public String getKey() {
        return "FFNN";
    }

    @Override
    public String getName() {
        return "FFNN";
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
        for (PartedPath path : candidatePaths)
            path.setMetric(network.getRegeneratorMetricValue() * (path.getNeededRegeneratorsCount()) + path.getMetric());
        for (int i = 0; i < candidatePaths.size(); i++)
            if (candidatePaths.get(i).getMetric() < 0) {
                candidatePaths.remove(i);
                i--;
            }
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
        } catch (Exception e) {
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
        try {
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("currXCord", data.currXCord);
            jsonObject.put("xCord1", data.xCord1);
            jsonObject.put("xCord2", data.xCord2);
            jsonObject.put("currYCord", data.currYCord);
            jsonObject.put("yCord1", data.yCord1);
            jsonObject.put("yCord2", data.yCord2);
            jsonObject.put("regenerators1", data.regenerators1);
            jsonObject.put("regenerators2", data.regenerators2);
            jsonObject.put("occupiedSpectrum1", data.occupiedSpectrum1);
            jsonObject.put("occupiedSpectrum2", data.occupiedSpectrum2);
            jsonObject.put("volume", data.volume);
            jsonObject.put("correctPath", data.correctPath);
            File file = new File("trainingData.json");
            if (file.exists()) {
                FileReader fileReader = new FileReader(file);
                JSONParser jsonParser = new JSONParser();
                JSONArray oldArray = (JSONArray) jsonParser.parse(fileReader);
                if (!oldArray.contains(jsonObject)) oldArray.add(jsonObject);
                Files.write(Paths.get(file.getAbsolutePath()), oldArray.toJSONString().getBytes());
            } else {
                JSONArray array = new JSONArray();
                file.createNewFile();
                array.add(jsonObject);
                Files.write(Paths.get(file.getAbsolutePath()), array.toJSONString().getBytes());
            }
        } catch (Exception ignored) {
            System.out.println("Failed to write the training data");
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
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .learningRate(0.05)
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(11)
                        .nOut(10)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU) //First hidden layer
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX) //Output layer
                        .lossFunction(LossFunctions.LossFunction.SQUARED_HINGE)
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
    public double currXCord;
    public double xCord1;
    public double xCord2;
    public double currYCord;
    public double yCord1;
    public double yCord2;
    public int regenerators1;
    public int regenerators2;
    public double occupiedSpectrum1;
    public double occupiedSpectrum2;
    public int volume;
    public int correctPath;

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
