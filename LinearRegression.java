import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class LinearRegression{

    public double arr_sum(double[] a){
        double sum = 0.0;
        for (int i=0; i<a.length; i++){
            sum += a[i];
        }
        return sum;
    }

    public double weight_arr_sum(double[] a, double w){
        double msum = 0.0;
        for (int i=0; i<a.length; i++){
            msum  = msum + (w * a[i]);
        }
        return msum;
    }

    public double[] bias_weight_sum(double[] a, double w, double b){
        int n = a.length;
        double[] sum = new double[n];
        for (int i=0;i<n;i++){
            double pred = w * a[i] + b;
            sum[i] = pred; 
        }
        return sum;
    }

    public double[] arr_minus(double[] a, double[] b){
        int n = a.length;
        double[] sum = new double[n];
        for (int i=0;i<n && i<b.length;i++){
            double pred = a[i] - b[i];
            sum[i] = pred; 
        }
        return sum;
    }

    public double[] arr_plus(double[] a, double[] b){
        int n = a.length;
        double[] sum = new double[n];
        for (int i=0;i<n;i++){
            double pred = a[i] + b[i];
            sum[i] = pred; 
        }
        return sum;
    }

    public double[] arr_square(double[] a){
        int n = a.length;
        double[] sum = new double[n];
        for (int i=0; i<n; i++){
            double pred = Math.pow(a[i], 2);
            sum[i] = pred;
        }
        return sum;
    }

    public double[] arr_arr_mul(double[] a, double[] b){
        int n = a.length;
        double[] sum = new double[n];
        for (int i=0; i<n; i++){
            double pred = a[i]*b[i];
            sum[i] = pred;
        }
        return sum;
    }

    public double cost(double[] X, double[] Y, double w, double b){
        double mse = 0.0;
        int m = Y.length;
        double[] predictions = bias_weight_sum(X, w, b);
        double[] error = arr_minus(predictions, Y);
        mse = (1.0 / (2.0 * m))*arr_sum(arr_square(error));
        return mse;
    }

    public double[] gradient_descent(double[] X, double[] Y, double w, double b, double initial_a, double decay_rate, double epochs){
        int m = Y.length;
        double weight_find=w;
        double bias_find=b;
        // double prev_cost = Double.MAX_VALUE; // Start with a very high cost

        for (int i=0;i<epochs;i++){
             // Dynamically adjust learning rate with decay
            double a = initial_a / (1 + decay_rate * i); 

            double[] predictions = bias_weight_sum(X, weight_find, bias_find);
            double[] error = arr_minus(predictions, Y);

            double d_weight = (1.0/m) * arr_sum(arr_arr_mul(error, X));
            double d_bias = (1.0/m) * arr_sum(error);

            weight_find = weight_find - (a*d_weight);
            bias_find = bias_find - (a*d_bias);

            // double current_cost = cost(X, Y, weight_find, bias_find);

            System.out.println("Epoch = "+(i+1)+" cost = " + cost(X, Y, weight_find, bias_find) + " error = "+ arr_sum(error));

            // if (Math.abs(prev_cost - current_cost) < 0.0001) {
            //     System.out.println("Converged at epoch " + (i + 1));
            //     break;
            // }

            // prev_cost = current_cost; // Update previous cost for the next iteration
        }
        double[] arr = {weight_find, bias_find};
        return arr;
    }

    public static double predict(double x, double w, double b){
        return w*x+b;
    }

     public static double[] readCSVColumn(String filePath, String columnName) {
        ArrayList<Double> values = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String headerLine = br.readLine(); // Read header
            String[] headers = headerLine.split(",");

            // Find index of required column
            int columnIndex = -1;
            for (int i = 0; i < headers.length; i++) {
                if (headers[i].trim().equalsIgnoreCase(columnName)) {
                    columnIndex = i;
                    break;
                }
            }

            if (columnIndex == -1) {
                System.out.println("Column " + columnName + " not found!");
                return new double[0];
            }

            // Read data
            String line;
            while ((line = br.readLine()) != null) {
                String[] columns = line.split(",");
                if (columnIndex < columns.length) {
                    try {
                        values.add(Double.parseDouble(columns[columnIndex].trim()));
                    } catch (NumberFormatException e) {
                        System.out.println("Skipping invalid value: " + columns[columnIndex]);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        return values.stream().mapToDouble(Double::doubleValue).toArray();
    }

    public static void main(String[] args) throws IOException {

        String filePath = "secret.txt"; // File name

        
        String content = new String(Files.readAllBytes(Paths.get(filePath))); // Read file into String
        System.out.println("File Content:\n" + content); // Print content
        

        double[] X = readCSVColumn(content, "LotArea");
        double[] Y = readCSVColumn(content, "SalePrice");

        System.out.println(Arrays.toString(X));
        System.out.println(Arrays.toString(Y));

        // double[] X= {100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0};
        // double[] Y= {150000.0, 240000.0, 350000.0, 460000.0, 570000.0, 660000.0, 750000.0, 860000.0, 950000.0, 1560000.0};
        double[] VX = new double[X.length];
        // double[] VY = new double[X.length];
        // for(int i=0; i<X.length; i++){
        //     VX[i] = X[i]/1000.0;
        // }
        double minX = Arrays.stream(X).min().getAsDouble();
        double maxX = Arrays.stream(X).max().getAsDouble();
        for (int i = 0; i < X.length; i++) {
            VX[i] = (X[i] - minX) / (maxX - minX);
        }
        // Finding min and max for Y
        double minY = Arrays.stream(Y).min().getAsDouble();
        double maxY = Arrays.stream(Y).max().getAsDouble();

        // Scaling Y values
        double[] scaledY = new double[Y.length];
        for (int i = 0; i < Y.length; i++) {
            scaledY[i] = (Y[i] - minY) / (maxY - minY);
        }
        Random random = new Random();
        // double weight = 0.0 + (1.0 - 0.0) * random.nextDouble();
        // double bias = 0.0 + (1.0 - 0.0) * random.nextDouble();
        // initializing weights to small random values closer to zero, e.g., -0.1 to 0.1, for better convergence.
        double weight = -0.1 + (0.2) * random.nextDouble();
        double bias = -0.1 + (0.2) * random.nextDouble();

        System.out.println("Weight = "+ weight);
        System.out.println("Bias = "+ bias);
        double a = 0.01;
        double decay_rate = 0.01;
        // double epochs = Y.length;
        double epochs = Y.length;
        LinearRegression lr = new LinearRegression();
        double arr[] = lr.gradient_descent(VX, scaledY, weight, bias, a, decay_rate, epochs);
        weight = arr[0];
        bias = arr[1];
        System.out.println(weight+ " "+ bias);

        Scanner sc=new Scanner(System.in);

        System.out.print("Enter the value you want to predict: ");
        double userInput = sc.nextDouble();
        double x = (userInput - minX) / (maxX - minX); 

        double scaledPrediction = predict(x, weight, bias); 
        double actualPrediction = scaledPrediction * (maxY - minY) + minY;

        System.out.println("Predicted price: " + actualPrediction);

        for(int i=0; i<X.length && i<Y.length; i++){
            if (X[i] == userInput){
                System.out.println(X[i] +"'s Actual Price = "+ Y[i]); //8450
                System.out.println("Difference(error) = " + Math.abs(actualPrediction-Y[i]));
                break;
            }
            continue;
        }

        sc.close();
    }
}