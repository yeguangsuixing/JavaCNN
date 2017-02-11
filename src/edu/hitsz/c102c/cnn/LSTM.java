import java.util.ArrayList;
import java.util.List;


public class LSTM {

	public static String join(int[] x){
		StringBuilder sb = new StringBuilder();
		sb.append(x[0]);
		for(int i = 1; i < x.length; i ++){sb.append(",");sb.append(x[i]);}
		return sb.toString();
	}
	public static int[] reversed(int[] x){
		int[] r = new int[x.length];
		for(int i = 0; i < x.length; i++){
			r[i] = x[x.length-1-i];
		}
		return r;
	}

	public static double[][] copy(double[][] x){
		double[][] r = new double[x.length][x[0].length];
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < x[0].length; j ++){
				r[i][j] = x[i][j];
			}
		}
		return r;
	}
	public static double[][] T(double[][] x){
		double[][] r = new double[x[0].length][x.length];
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < x[0].length; j ++){
				r[j][i] = x[i][j];
			}
		}
		return r;
	}
	public static double[][] sigmoid(double[][] x){
		double[][] r = new double[x.length][x[0].length];
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < x[0].length; j ++){
				r[i][j] = 1/(1+Math.exp(-x[i][j]));
			}
		}
		return r;
		//return 1/(1+Math.exp(-x));
	}
	public static double[][] add(double[][] x, double[][] b){
		double[][] r = new double[x.length][x[0].length];
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < x[0].length; j ++){
				r[i][j] = x[i][j] + b[i][j];
			}
		}
		return r;
	}
	public static double[][] minus(double[][] x, double[][] b){
		double[][] r = new double[x.length][x[0].length];
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < x[0].length; j ++){
				r[i][j] = x[i][j] - b[i][j];
			}
		}
		return r;
	}
	public static double[][] sigmoid_output_to_derivative(double[][] x){
		double[][] r = new double[x.length][x[0].length];
		for(int i = 0; i < x.length; i++){
			for(int j = 0; j < x[0].length; j ++){
				r[i][j] = x[i][j]*(1-x[i][j]);
			}
		}
		return r;//x*(1-x);
	}
	public static double[][] multiple(double[][] a, double b){
		double[][] r = new double[a.length][a[0].length];
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < a[0].length; j++){
				r[i][j] = a[i][j]*b;
			}	
		}
		return r;
	}
	public static double[][] dot(double[][] a, double[][] b){
		double[][] r = new double[a.length][b[0].length];
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < b[0].length; j++){
				r[i][j] = 0;
				for(int k = 0; k < a[0].length; k ++){
					r[i][j] += a[i][k]*b[k][j];
				}
			}	
		}
		return r;
	}
	public static double[][] multiple(double[][] a, double[][] b){
		double[][] r = new double[a.length][b[0].length];
		for(int i = 0; i < a.length; i++){
			for(int j = 0; j < b[0].length; j++){
				r[i][j] = a[i][j]*b[i][j];
			}	
		}
		return r;
	}
	
	public static void main(String[] args){
		
		//training data set generation
		int binary_dim = 8;
		 
		int largest_number = 1<<binary_dim;//(int)Math.pow(2, binary_dim);
		int[][] int2binary = new int[largest_number][binary_dim];//{};
		for(int i = 0; i < largest_number; i ++){// i in range(largest_number):
			for(int j = 0; j < 8; j++){
				int2binary[i][j] = (i>>(7-j)) & 1;
			}
		}
		 
		//input variables
		double alpha = 0.1;
		int input_dim = 2;//dimension
		int hidden_dim = 10;
		int output_dim = 1;
		 
		//initialize neural network weights
		double[][] synapse_0 = new double[input_dim][hidden_dim];
		//2*np.random.random((input_dim,hidden_dim)) - 1
		double[][] synapse_1 = new double[hidden_dim][output_dim];
		//2*np.random.random((hidden_dim,output_dim)) - 1
		double[][] synapse_h = new double[hidden_dim][hidden_dim];
		for(int i=0;i<input_dim;i++)for(int j=0;j<hidden_dim;j++)
			synapse_0[i][j] = 2*Math.random()-1;
		for(int i=0;i<hidden_dim;i++)for(int j=0;j<output_dim;j++)
			synapse_1[i][j] = 2*Math.random()-1;
		for(int i=0;i<hidden_dim;i++)for(int j=0;j<hidden_dim;j++)
			synapse_h[i][j] = 2*Math.random()-1;
		//2*np.random.random((hidden_dim,hidden_dim)) - 1
		 
		double[][] synapse_0_update = new double[input_dim][hidden_dim];
		//np.zeros_like(synapse_0)
		double[][] synapse_1_update = new double[hidden_dim][output_dim];
		//np.zeros_like(synapse_1)
		double[][] synapse_h_update = new double[hidden_dim][hidden_dim];
		//np.zeros_like(synapse_h)
		 
		//training logic
		for(int j = 0; j<10000; j++){
		    //generate a simple addition problem (a + b = c)
		    int a_int = (int)(Math.random()*(largest_number/2));
		    //np.random.randint(largest_number/2) # int version
		    int[] a = int2binary[a_int]; // binary encoding
		 
		    int b_int = (int)(Math.random()*(largest_number/2));
		    //np.random.randint(largest_number/2) // int version
		    int[] b = int2binary[b_int]; // binary encoding
		 
		    //true answer
		    int c_int = a_int + b_int;
		    int[] c = int2binary[c_int];
		 
		    //where we'll store our best guess (binary encoded)
		    int[] d = new int[c.length];//np.zeros_like(c)
		 
		    double overallError = 0f;
		 
		    List<double[][]> layer_2_deltas = new ArrayList<double[][]>();//list()
		    List<double[][]> layer_1_values = new ArrayList<double[][]>();//list()
		    layer_1_values.add(new double[1][hidden_dim]);//np.zeros(hidden_dim));
		 
		    //moving along the positions in the binary encoding
		    for(int position = 0; position < binary_dim; position ++){
		 
		        //generate input and output
		    		int ai = a[binary_dim - position - 1],
		    				bi = b[binary_dim - position - 1],
		    				ci = c[binary_dim - position - 1];
		        double[][] X = new double[][]{{ai,bi}};//np.array([[ai,bi]]);
		        double[][] y = new double[][]{{ci}};//y = np.array([[ci]]).T;
		 
		        //hidden layer (input ~+ prev_hidden)
		        double[][] layer_1 = sigmoid(
		        		add(
			        		dot(X,synapse_0),
			        		dot(layer_1_values.get(layer_1_values.size()-1),synapse_h)
		        		)
		        	);
		 
		        //output layer (new binary representation)
		        double[][] layer_2 = sigmoid(dot(layer_1,synapse_1));
		 
		        //did we miss?... if so by how much?
		        double[][] layer_2_error = minus(y, layer_2);
		        layer_2_deltas.add(
		        		multiple((layer_2_error),sigmoid_output_to_derivative(layer_2))
		        	);
		        overallError += Math.abs(layer_2_error[0][0]);
		 
		        //decode estimate so we can print it out
		        d[binary_dim - position - 1] = (int)Math.round(layer_2[0][0]);
		 
		        //store hidden layer so we can use it in the next timestep
		        layer_1_values.add(copy(layer_1));
		    }
		 
		    double[][] future_layer_1_delta = new double[1][hidden_dim];//np.zeros(hidden_dim)
		    
		 
		    for(int position = 0; position < binary_dim; position ++){
		 
		        double[][] X = new double[][]{{a[position],b[position]}};
		        //np.array([[a[position],b[position]]]);
		        double[][] layer_1 = layer_1_values.get(layer_1_values.size()-position-1);
		        double[][] prev_layer_1 = layer_1_values.get(layer_1_values.size()-position-2);
		 
		        // error at output layer
		        double[][] layer_2_delta = layer_2_deltas.get(layer_2_deltas.size()-position-1);
		        // error at hidden layer
		        double[][] layer_1_delta = multiple(
		        		add(
		        				dot(future_layer_1_delta, 	T(synapse_h)),
		        				dot(layer_2_delta, 			T(synapse_1))
		        		),
		        		sigmoid_output_to_derivative(layer_1)
		        );
		        // let's update all our weights so we can try again
		        synapse_1_update = add(synapse_1_update, dot(T(layer_1), layer_2_delta));
		        synapse_h_update = add(synapse_h_update, dot(T(prev_layer_1),layer_1_delta));
		        synapse_0_update = add(synapse_0_update, dot(T(X), layer_1_delta));
		 
		        future_layer_1_delta = layer_1_delta;
		    }
		    synapse_0 =add(synapse_0, multiple(synapse_0_update, alpha));
		    synapse_1 =add(synapse_1, multiple(synapse_1_update, alpha));
		    synapse_h =add(synapse_h, multiple(synapse_h_update, alpha));   
		 
		    synapse_0_update = multiple(synapse_0_update, 0);
		    synapse_h_update = multiple(synapse_h_update, 0);
		    synapse_1_update = multiple(synapse_1_update, 0);
		 
		    // print out progress
		    if(j % 1000 == 0 || j == 20000-1){
		        System.out.println("j:" + j);
		        System.out.println("Error:" + overallError);
		        System.out.println("Pred:" + join(d));
		        	System.out.println("True:" + join(c));
		        int out = 0;
		        int[] dr = reversed(d);
		        for(int i = 0; i < dr.length; i++){// index,x in enumerate(){
		            out += dr[i] * (1<<i);//x*pow(2,index);
		        }
		        System.out.println(a_int + " + " + b_int + " = " + out);
		        System.out.println("------------");
		    }
		}
	}
}
