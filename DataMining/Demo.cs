using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining
{
    class Demo
    {
        private static double[,] learningData = new double[,]
            {
                {255,0,0}, {210,45,10}, {121,12,100}, {47,12,7}, {180,57,100},
                {125,10,81}, {90,35,07}, {187,112,141}, {147,13,7}, {189,57,100},
                {215,3,149}, {210,145,15}, {129,112,101}, {147,20,71}, {188,157,12},
                {175,10,61}, {190,36,17}, {187,102,101}, {144,13,7}, {199,57,110},

                {147,254,010}, {11,154,101}, {147,254,210}, {105,124,100}, {14,87,14},
                {127,204,018}, {41,187,11}, {10,124,50}, {105,224,70}, {14,187,14},
                {129,279,017}, {51,159,102}, {141,253,212}, {101,127,109}, {18,81,11},
                {129,209,019}, {44,189,12}, {15,126,56}, {115,223,71}, {18,189,15},

                {147,54,170}, {144,24,207}, {37,74,114}, {14,51,87}, {1,74,104},
                {45,154, 223}, {34,54,187}, {32,54,94}, {114,15,187}, {12,74,214},
                {147,54,178}, {144,21,209}, {32,77,119}, {15,21,87}, {5,79,109},
                {44,153, 222}, {35,55,188}, {33,55,96}, {115,16,185}, {11,72,218},
            };
        private static string[] learningClasses = new string[]
            {
                "red", "red", "red", "red", "red",
                "red", "red", "red", "red", "red",
                "red", "red", "red", "red", "red",
                "red", "red", "red", "red", "red",
                
                "green", "green", "green", "green", "green",
                "green", "green", "green", "green", "green",
                "green", "green", "green", "green", "green",
                "green", "green", "green", "green", "green",
                
                "blue", "blue", "blue", "blue", "blue",
                "blue", "blue", "blue", "blue", "blue",
                "blue", "blue", "blue", "blue", "blue",
                "blue", "blue", "blue", "blue", "blue",
            };
        private static double[,] dataToPredict = new double[,]
            {
            };

        //Variables used to show training progression
        private static int treeCnt = 0;
        private static int modulo = 100;

        private static Random r = new Random();

        static void Main(String[] args)
        {
            //Initialize a new motor
            RandomForest RFmotor = new RandomForest();
            //Enable or disable suspect-detection. Elements with less than 30% of prediction will be classed as "suspect"
            RFmotor.enableOtherClassCreation(30, "suspect");
            RFmotor.disableOtherClassCreation();
            //Do something when a tree is created
            RFmotor._TreeCreated += RFmotor__TreeCreated;
            
            do
            {
                Console.Clear();

                //Initialize random data to predict
                generateDataToPredict(10);
                //Initialize random learning set
                generateLearningData(50);

                //Evaluate the learning (optional, usefull when creating a new learning)
                modulo = 100;
                Console.Write("Training : ");
                ConfusionData cd = RFmotor.evaluateCrossValidation(5, learningData, learningClasses, 500, 100);
                Console.WriteLine();
                displayConfusionData(cd);
                Console.WriteLine();

                //Train the motor with learning data
                modulo = 10;
                Console.Write("Training : ");
                RFmotor.train(learningData, learningClasses, 500, 100);
                Console.WriteLine();

                //Predict sample's class
                Console.WriteLine("Predicted data : ");
                for (int i = 0; i < dataToPredict.GetLength(0); i++)
                {
                    double[] row = new double[dataToPredict.GetLength(1)];
                    for (int j = 0; j < dataToPredict.GetLength(1); j++)
                        row[j] = dataToPredict[i,j];
                    Prediction predicted = RFmotor.predict(row);
                    displayPrediction(predicted, row);
                }
            } while (Console.ReadKey().Key != ConsoleKey.Escape);
            Console.ReadLine();
        }

        private static void generateLearningData(int linesPerCategory)
        {
            learningData = new double[linesPerCategory * 3, 3];
            learningClasses = new string[linesPerCategory * 3];
            for (int i = 0; i < 3 * linesPerCategory; i++)
            {
                String category = i < linesPerCategory ? "red" : i < linesPerCategory * 2 ? "green" : "blue";

                List<double> vals = new List<double>();
                for (int k = 0; k < 3; k++)
                    vals.Add(r.Next(0, 256));
                vals = vals.OrderByDescending(n => n).ToList();

                if (category == "red")
                {
                    learningData[i, 0] = vals[0];
                    learningData[i, 1] = vals[1];
                    learningData[i, 2] = vals[2];
                }
                else if (category == "green")
                {
                    learningData[i, 0] = vals[1];
                    learningData[i, 1] = vals[0];
                    learningData[i, 2] = vals[2];
                }
                else if (category == "blue")
                {
                    learningData[i, 0] = vals[2];
                    learningData[i, 1] = vals[1];
                    learningData[i, 2] = vals[0];
                }
                learningClasses[i] = category;
            }
        }

        static void RFmotor__TreeCreated(object sender, EventArgs e)
        {
            treeCnt++;
            if (treeCnt % modulo == 0)
                Console.Write("#");
        }

        private static void generateDataToPredict(int lines)
        {
            dataToPredict = new double[lines, learningData.GetLength(1)];
            for (int i = 0; i < lines; i++)
            {
                for (int j = 0; j < learningData.GetLength(1); j++)
                {
                    dataToPredict[i, j] = r.Next(0, 255);
                }
            }
        }

        private static void displayPrediction(Prediction predicted, double[] data)
        {
            Console.Write("Data : {");
            for(int i=0;i<data.Length;i++)
            {
                Console.Write(data[i].ToString("000"));
                if (i != data.Length - 1)
                    Console.Write(", ");
                else
                    Console.Write("}");
            }
            Console.Write(" - Predicted class : " + predicted.maxClass + " (" + predicted.treePredictedClass + "/" + predicted.totalTrees+")");

            if (data.Length == 3)
            {
                String realClass = "";
                double max = Math.Max(data[0], Math.Max(data[1], data[2]));
                realClass = max == data[0] ? "red" : max == data[1] ? "green" : "blue";
                if (realClass != predicted.maxClass)
                    Console.Write(" !FAIL");
            }
            Console.WriteLine();
        }

        private static void displayConfusionData(ConfusionData cd)
        {
            Console.WriteLine("Confusion data from learning set : ");
            int maxLengthName = cd.UniqueCat.Aggregate("", (max, cur) => max.Length > cur.Length ? max : cur).Length;
            for (int i = 0; i < cd.data.Count; i++)
            {
                Console.Write(cd.UniqueCat[i]);
                for(int k =0; k < maxLengthName - cd.UniqueCat[i].Length; k++)
                    Console.Write(" ");
                Console.Write(" | ");

                int wellPredicted = cd.data[i][i];
                int badPredicted = 0;
                for (int j = 0; j < cd.data[0].Count; j++)
                {
                    Console.Write(cd.data[i][j].ToString("000") + " | ");
                    if (j != i)
                        badPredicted += cd.data[i][j];
                }
                if (wellPredicted + badPredicted == 0)
                    Console.WriteLine();
                else
                    Console.WriteLine("("+(wellPredicted*100/(wellPredicted+badPredicted)).ToString("00.0") +"%)");
            }
        }
    }
}
