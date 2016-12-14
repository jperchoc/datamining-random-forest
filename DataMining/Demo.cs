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

                {147,254,010}, {11,154,101}, {147,254,210}, {105,124,100}, {14,87,14},
                {127,204,018}, {41,187,11}, {10,124,50}, {105,224,70}, {14,187,14},

                {147,54,170}, {144,24,207}, {37,74,114}, {14,51,87}, {1,74,104},
                {147,54,170}, {144,24,207}, {37,74,114}, {14,51,87}, {1,74,104},
            };
        private static string[] learningClasses = new string[]
            {
                "red  ", "red  ", "red  ", "red  ", "red  ",
                "red  ", "red  ", "red  ", "red  ", "red  ",
                
                "green", "green", "green", "green", "green",
                "green", "green", "green", "green", "green",
                
                "blue ", "blue ", "blue ", "blue ", "blue ",
                "blue ", "blue ", "blue ", "blue ", "blue ",
            };
        private static double[,] dataToPredict = new double[,]
            {
                
                {201,42,20},
                {141,200,54},
                {74,12,117},
            };

        static void Main(String[] args)
        {
            //Initialize a new motor
            RandomForest RFmotor = new RandomForest();
            //Enable or disable suspect-detection. Elements with less than 30% of prediction will be classed as "suspect"
            RFmotor.enableOtherClassCreation(30, "suspect");
            RFmotor.disableOtherClassCreation();
            //Do something when a tree is created
            RFmotor._TreeCreated += ((sender, e) => {  /* place code here */ });
            do
            {
                Console.Clear();
                //Evaluate the learning (optional, usefull when creating a new learning)
                ConfusionData cd = RFmotor.evaluateCrossValidation(5, learningData, learningClasses, 500, 100);
                displayConfusionData(cd);
                //Train the motor with learning data
                RFmotor.train(learningData, learningClasses, 500, 100);
                //Predict sample's class
                Console.WriteLine();
                Console.WriteLine("Predicted data : ");
                for (int i = 0; i < dataToPredict.GetLength(0); i++)
                {
                    double[] row = new double[dataToPredict.GetLength(1)];
                    for (int j = 0; j < dataToPredict.GetLength(1); j++)
                        row[j] = dataToPredict[i, j];
                    Prediction predicted = RFmotor.predict(row);
                    displayPrediction(predicted, row);
                }
            } while (Console.ReadKey().Key != ConsoleKey.Escape);
            Console.ReadLine();
        }

        private static void displayPrediction(Prediction predicted, double[] data)
        {
            Console.Write("Data : {");
            for(int i=0;i<data.Length;i++)
            {
                Console.Write(data[i]);
                if (i != data.Length - 1)
                    Console.Write(",");
                else
                    Console.Write("}");
            }
            Console.WriteLine(" - Predicted class : " + predicted.maxClass + " (" + predicted.treePredictedClass + "/" + predicted.totalTrees+")");
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
