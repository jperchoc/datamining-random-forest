/************************************************/
/* Project:        RandomForest                 */
/* File:           RandomForest.cs              */
/* Author:         Jonathan Perchoc             */
/* Company:        IFREMER                      */
/* Created:        27/03/2015                   */
/* Last modified:  18/01/2016                   */
/************************************************/
/* Description: Classe permettant de faire le   */
/*              training, la prédiction et      */
/*              l'évaluation du learning set    */
/*              via cross validation.           */
/*  inspiré de http://semanticquery.com/archive/semanticsearchart/downloads/RFtest.zip
/************************************************/
/* 27/03/2015 : Mise en place de la détection   */
/*              de suspects                     */
/* 18/01/2016 : Corrections d'errreurs, ajout   */
/*              de nombreux commentaires,       */
/*              optimisation via linq           */
/* 03/02/2016 : Ajout du pourcentage d'arbres   */
/*              prédisant la classe, changement */
/*              du type de retour de la méthode */
/*              predict                         */
/* 15/03/2016 : Mise en place du multithreading */
/************************************************/
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataMining
{
    public class RandomForest
    {
        public String CUT_CLASS_NAME = "Cut_Objects";
        protected List<RandomForestNode> _forest;
        protected List<String> _categories;
        protected int _tailleDeSousEchantillonAleatoire;
        protected Random _randomGenerator = new Random();
        protected int _numberRandomParameters;
        protected int _nParameters;
        protected double[][] _learning;
        protected String[] _learningCategories;
        public event EventHandler _TreeCreated;
        protected bool _createOtherClass;
        protected double _percentageOther;
        protected String _otherClassName = "other";
        public bool PredictCuts { get; set; }
        public int CutTagIndex { get; set; }

        //Lock random pour multithread
        object syncLock = new object();

        public bool moyennageClassifier = true;

        public String OtherClassName
        {
            get { return _otherClassName; }
        }

        /// <summary>
        /// Active la création de la classe "other".
        /// </summary>
        /// <param name="percentage">Pourcentage de votes minimum pour ne pas être attribué à la classe "other"</param>
        public void enableOtherClassCreation(int percentage, String otherClassName = "other")
        {
            this._createOtherClass = true;
            this._percentageOther = (double)(percentage / 100.0);
            this._otherClassName = otherClassName;
        }
        /// <summary>
        /// Désactive la création de la classe "other".
        /// </summary>
        public void disableOtherClassCreation()
        {
            this._createOtherClass = false;
        }
        /// <summary>
        /// Crée la forêt d'arbre de prédictions à partir du learning passé en paramètre.                                          
        /// </summary>
        /// <param name="samples">Matrice d'échantillons de learning</param>
        /// <param name="samplesCategories">Tableau contenant la catégorie de chacun des échantillons du learning</param>
        /// <param name="forestSize">Nombre d'arbres de prédiction à créer</param>
        /// <param name="percentParameters">Pourcentage de paramètres a prendre en compte pour chaque arbre. Valeur par défaut : log2 M+1</param>
        public void train(double[,] samples, String[] samplesCategories, int forestSize, int percentParameters = 0)
        {
            //On initialise la liste d'arbres décisionnels
            this._forest = new List<RandomForestNode>();
            //On récupère le nombre de paramètres
            this._nParameters = samples.GetLength(1);
            //On récupère les différentes catégories du learning set dans une liste   
            this._categories = samplesCategories.Distinct().ToList();
            //On définit la taille du sous-échantillon aléatoire égal à 64% du nombre d'échantillons du learning
            this._tailleDeSousEchantillonAleatoire = (int)(0.64 * (double)(samplesCategories.GetLength(0)));
            //On définit le nombre de paramètres aléatoires à tirer
            if (percentParameters == 0)
                _numberRandomParameters = (int)Math.Sqrt(_nParameters);
            //_percentageParameters = (double)(Math.Log(_nParameters + 1, 2) / _nParameters);
            else
                this._numberRandomParameters = (int)(percentParameters / 100.0 * _nParameters);
            //On récupère les échantillons du learning sous forme de tableau 2D jagged
            this._learning = ToJaggedArray<double>(samples);
            //On récupère les catégories de chaque échantillon
            this._learningCategories = samplesCategories;
            //On crée la forêt d'arbres décisionnels
            populateForest(forestSize, samples);
        }
        /// <summary>
        /// Prédit la classe de l'échantillon passé en paramètre. Un learning doit avoir été entraîné via la méthode train.
        /// </summary>
        /// <param name="sample">Paramètres de l'échantillon à prédire</param>
        /// <returns>Classe prédite</returns>
        public Prediction predict(double[] sample)
        {
            Prediction p = new Prediction();
            //Checking
            if (_learning == null)
                throw new InvalidOperationException("Aucun learning n'a été défini.");
            if (sample.Count() != _nParameters)
                throw new InvalidOperationException("L'echantillon et le learning n'ont pas le même nombre de paramètres");
            //Prediction
            if (!PredictCuts && sample[CutTagIndex] == 1)
            {
                p.maxClass = CUT_CLASS_NAME;
                p.predictedClass = CUT_CLASS_NAME;
                p.totalTrees = this._forest.Count;
                p.treePredictedClass = p.totalTrees;
            }
            else
            {
                //Compteur d'occurence des catégories prédites
                int[] counter = new int[_categories.Count()];
                for (int c = 0; c < _categories.Count(); ++c)
                    counter[c] = 0;
                //Pour chaque arbre de la forêt, on prédit la catégorie
                foreach (RandomForestNode tree in _forest)
                {
                    String preditedCategory = this.getCategoryFromTree(tree, sample);
                    counter[_categories.IndexOf(preditedCategory)]++;
                }
                p.maxClass = this._categories[counter.ToList().IndexOf(counter.Max())];
                p.treePredictedClass = counter.Max();
                p.totalTrees = this._forest.Count();
                //On renvoie la catégorie prédite
                if (this._createOtherClass && (counter.Max() < this._forest.Count * this._percentageOther))
                    p.predictedClass = this._otherClassName;
                else
                {
                    p.predictedClass = this._categories[counter.ToList().IndexOf(counter.Max())];
                }
            }
            return p;
        }

        public ConfusionData evaluateCrossValidation(int itterations, double[,] samples, String[] samplesCategories, int forestSize, int percentParameters = 0)
        {
            int error = 0;
            ConfusionData cd = new ConfusionData();
            var categories = samplesCategories.Distinct().ToList();
            if (this._createOtherClass)
                categories.Add(this._otherClassName);
            if(!PredictCuts)
                categories.Add(CUT_CLASS_NAME);
            cd.UniqueCat = categories;

            //On crée deux sous tableaux constitués d'échantillons tirés aléatoirement
            double[][] samplesArray = ToJaggedArray<double>(samples);
            for (int itt = 0; itt < itterations; itt++)
            {
                //On crée des index aléatoires pour créer les tableaux
                int[] indexes = getRandomSamplesOutOfTrainingSet(samplesArray.GetLength(0), (int)(samplesArray.Length * 0.6));

                int sizeA = (int)(samplesArray.Length * 0.6);
                int sizeB = (int)(samplesArray.Length * 0.4);
                while (sizeA + sizeB != samplesArray.Length)
                    sizeB++;

                double[][] A = new double[sizeA][];
                double[][] B = new double[sizeB][];

                int j = 0;
                int k = 0;
                String[] ACat = new string[sizeA];
                String[] BCat = new string[sizeB];
                for (int i = 0; i < samplesArray.Length; i++)
                {
                    if (indexes.Contains(i))
                    {
                        ACat[j] = samplesCategories[i];
                        A[j] = samplesArray[i];
                        j++;
                    }
                    else
                    {
                        BCat[k] = samplesCategories[i];
                        B[k] = samplesArray[i];
                        k++;
                    }
                }

                //Les deux tableaux sont crées
                train(To2DArray<double>(A), ACat, forestSize, percentParameters);
                for (int s = 0; s < B.Length; s++)
                {
                    Prediction pred = this.predict(B[s]);
                    if (pred.predictedClass != BCat[s])
                        error++;

                    cd.data[cd.UniqueCat.IndexOf(BCat[s])][cd.UniqueCat.IndexOf(pred.predictedClass)]++;

                    if (pred.predictedClass != this.OtherClassName)
                    {
                        var percentage = cd.pourcentages.First(p => p.categorie == pred.predictedClass);
                        percentage.predicted += pred.treePredictedClass;
                        percentage.total += pred.totalTrees;
                    }
                }

                train(To2DArray<double>(B), BCat, forestSize, percentParameters);
                for (int s = 0; s < A.Length; s++)
                {
                    Prediction pred = this.predict(A[s]);
                    if (pred.predictedClass != ACat[s])
                        error++;
                    cd.data[cd.UniqueCat.IndexOf(ACat[s])][cd.UniqueCat.IndexOf(pred.predictedClass)]++;

                    if (pred.predictedClass != this.OtherClassName)
                    {
                        var percentage = cd.pourcentages.First(p => p.categorie == pred.predictedClass);
                        percentage.predicted += pred.treePredictedClass;
                        percentage.total += pred.totalTrees;
                    }
                }
            }
            // ((double)error / (samples.GetLength(0))) / (double)itterations * 100.0;
            return cd;
        }
        #region surcharges
        public string predict(int[] sample)
        {
            return this.predict(sample);
        }
        public string predict(float[] sample)
        {
            return this.predict(sample);
        }
        public string predict(byte[] sample)
        {
            return this.predict(sample);
        }
        public string predict(bool[] sample)
        {
            return this.predict(sample);
        }
        #endregion

        protected List<System.Threading.AutoResetEvent> _EventsWorkerCompleted = new List<System.Threading.AutoResetEvent>();

        protected void populateForest(int forestSize, double[,] samples)
        {
            _EventsWorkerCompleted.Clear();
            //On trouve combien de thread créer
            int nbThreads = 8;
            for (int t = 0; t < nbThreads; t++)
            {
                _EventsWorkerCompleted.Add(new System.Threading.AutoResetEvent(false));
                BackgroundWorker bg = new BackgroundWorker();
                bg.DoWork += ((a, b) =>
                {
                    int nThread = (int)b.Argument;
                    List<RandomForestNode> trees = new List<RandomForestNode>();
                    for (int i = 0; i < forestSize / nbThreads; i++)
                    {
                        RandomForestNode result = createTree(samples);
                        trees.Add(result);
                        OnTreeCreated(EventArgs.Empty);
                    }
                    lock (_forest)
                        _forest.AddRange(trees);
                    _EventsWorkerCompleted[nThread].Set();
                });
                bg.RunWorkerAsync(t);
            }

            foreach (var eventWait in _EventsWorkerCompleted)
                eventWait.WaitOne();
            //_workerCompleted.WaitOne();

            //On ajoute les arbres manquants
            List<RandomForestNode> treesa = new List<RandomForestNode>();
            int manquants = forestSize - this._forest.Count;
            for (int i = 0; i < manquants; i++)
            {
                treesa.Add(createTree(samples));
                OnTreeCreated(EventArgs.Empty);
            }
            lock (_forest)
                _forest.AddRange(treesa);
        }

        protected RandomForestNode createTree(double[,] samples)
        {
            //On récupère les index dans le tableau _learning des _tailleDeSousEchantillonAleatoire échantillons tirés aléatoirement
            int[] randomSamplesIndexesOutOfTraining = this.getRandomSamplesOutOfTrainingSet(_learning.GetLength(0), _tailleDeSousEchantillonAleatoire);
            //On récupère une liste numérotée de 0 à _nParameters
            List<int> param = new List<int>();
            for (int j = 0; j < this._nParameters; j++)
                param.Add(j);
            //On crée un noeud à partir des échantillons aléatoires via la factory
            RandomForestNode node = this.nodeFactory(randomSamplesIndexesOutOfTraining);
            //Si on n'est pas arrivé sur une feuille de l'arbre
            if (node.category == null)
            {
                //On crée les branches droites et gauche du noeud
                this.makeLeftAndRightNodes(node, randomSamplesIndexesOutOfTraining);
            }
            //On retourne le noeud
            return node;
        }
        /// <summary>
        /// Tire aléatoirement des échantillons uniques parmis ceux du learning set
        /// </summary>
        /// <returns></returns>
        protected int[] getRandomSamplesOutOfTrainingSet(int arraySize, int howmany)
        {
            HashSet<int> hash = new HashSet<int>();
            hash.Clear();
            int[] samplesIndexesInLearning = new int[howmany];
            int cnt = 0;
            do
            {
                int randomInt;
                lock (syncLock)
                {
                    randomInt = this._randomGenerator.Next(arraySize);
                }
                if (!hash.Contains(randomInt))
                {
                    samplesIndexesInLearning[cnt++] = randomInt;
                    hash.Add(randomInt);
                }
            } while (cnt < howmany);
            return samplesIndexesInLearning;
        }

        protected RandomForestNode nodeFactory(int[] samplesIndexesInLearning)
        {
            //On crée un objet node
            RandomForestNode node = new RandomForestNode();
            //Si il n'y a qu'un echantillon, on est sur une leaf
            if (samplesIndexesInLearning.Length == 1)
            {
                node.category = _learningCategories[(int)samplesIndexesInLearning[0]];
            }
            else //Sinon, on crée la condition
            {
                //On tire les paramètres aléatoires parmis tous les paramètres
                List<int> param = new List<int>();
                param.Clear();
                int cnt = 0;
                do
                {
                    int randomInt;
                    lock (syncLock)
                    {
                        randomInt = this._randomGenerator.Next(_nParameters);
                    }
                    if (!param.Contains(randomInt))
                    {
                        cnt++;
                        param.Add(randomInt);
                    }
                } while (cnt < _numberRandomParameters);
                //Ces paramètres aléatoires serviront à calculer le treshold
                node.paramThreshold = param;
                //On crée le classifier
                node.classifier = makeClassifier(samplesIndexesInLearning);
                //On calcule le threshold
                node.threshold = getThreshold(samplesIndexesInLearning, node);
            }
            return node;
        }

        protected void makeLeftAndRightNodes(RandomForestNode node, int[] samplesIndexesInLearning)
        {
            //On sépare l'échantillon en deux
            byte[] indicator = splitSample(node, samplesIndexesInLearning);
            //On récupère l'échantillon séparé dans des tableaux left et right
            int nLeftVectors = 0;
            foreach (byte b in indicator)
                if (b > 0)
                    ++nLeftVectors;
            int[] left = new int[nLeftVectors];
            int[] right = new int[indicator.Length - nLeftVectors];
            int cntL = 0;
            int cntR = 0;
            int cnt = 0;
            foreach (byte b in indicator)
            {
                if (b > 0)
                    left[cntL++] = samplesIndexesInLearning[cnt];
                else
                    right[cntR++] = samplesIndexesInLearning[cnt];
                ++cnt;
            }
            //On affecte les tableaux au noeud actuel après en avoir fait des noeuds
            node.left = nodeFactory(left);
            node.right = nodeFactory(right);
            //Si les nouveaux noeuds ne sont pas des feuilles, on leur crée leurs neouds gauche et droite par récursivité
            if (node.left.category == null)
                makeLeftAndRightNodes(node.left, left);
            if (node.right.category == null)
                makeLeftAndRightNodes(node.right, right);
        }
        /// <summary>
        /// Parcours un arbre jusqu'à récupérer la catégorie prédite
        /// </summary>
        /// <param name="node">Noeud à évaluer</param>
        /// <param name="data">Tableau à prédire</param>
        /// <returns>Catégorie prédite pour ce tableau</returns>
        protected String getCategoryFromTree(RandomForestNode node, double[] data)
        {
            if (node.category != null)
                return node.category;
            double f = getCosineSimilarity(data, node.classifier, node.paramThreshold);
            if (f > node.threshold)  // = ?
                return getCategoryFromTree(node.left, data);
            else
                return getCategoryFromTree(node.right, data);
        }
        protected double[] makeClassifier(int[] samplesIndexesInLearning)
        {
            //On compte le nombre de fois que chaque classe apparaît dans les échantillons passés en paramètres (histogramme)
            int[] classesHistogram = new int[_categories.Count];
            foreach (int x in samplesIndexesInLearning)
                ++classesHistogram[_categories.IndexOf(_learningCategories[x])];
            //On récupère la catégorie la plus représentée dans la liste d'échantillons
            int mostRepresentativeCategory = classesHistogram.ToList().IndexOf(classesHistogram.Max());
            //On récupère les échantillons correcpondants a la classe la plus représentative
            int[] vectors = samplesIndexesInLearning.Where(s => _categories.IndexOf(_learningCategories[s]) == mostRepresentativeCategory).ToArray();
            //On crée le classifier : pour chaque paramètre, on calcule la somme des valeurs de ce paramètre parmis les échantillons du tableau vectors
            double[] classifier = new double[_nParameters];
            for (int i = 0; i < _nParameters; ++i)
            {
                foreach (int x in vectors)
                    classifier[i] += _learning[x][i];
            }
            //test : moyennage du classifier
            if (moyennageClassifier)
                for (int k = 0; k < classifier.Length; k++)
                    classifier[k] /= vectors.Length;
            return classifier;
        }

        //Calcule la similarité cosinus de deux tableaux
        // https://en.wikipedia.org/wiki/Cosine_similarity
        //La similarité cosinus de deux tableaux donne un indice de proximité entre deux vecteurs
        /*
         *                          A.B             ∑AiBi
         * similarity = cos(θ) = ————————— = ———————————————————
         *                        ǁAǁ.ǁBǁ     √(∑Ai²) * √(∑Bi²)
         * */
        protected double getCosineSimilarity(double[] s1, double[] s2, List<int> parametres)
        {
            double product = 0.0;
            double sqr1 = 0.0;
            double sqr2 = 0.0;
            foreach (int i in parametres)
            {
                product += (double)(s1[i]) * (double)(s2[i]);
                sqr1 += (double)(s1[i]) * (double)(s1[i]);
                sqr2 += (double)(s2[i]) * (double)(s2[i]);
            }
            return product / (Math.Sqrt(sqr1) * Math.Sqrt(sqr2));
        }
        //Calcule le threshold d'un noeud en faisant la moyenne des similarités cosinus min et max
        protected double getThreshold(int[] samplesIndexesInLearning, RandomForestNode n)
        {
            double fmin = 1000.0;
            double fmax = -1.0;
            foreach (int x in samplesIndexesInLearning)
            {
                double f = getCosineSimilarity(this._learning[x], n.classifier, n.paramThreshold);
                if (f < fmin)
                    fmin = f;
                if (f > fmax)
                    fmax = f;
            }
            return (fmax + fmin) / 2.0;
        }
        protected byte[] splitSample(RandomForestNode node, int[] samplesIndexesInLearning)
        {
            byte[] indicator = new byte[samplesIndexesInLearning.Length];
            int cnt = 0;
            foreach (int x in samplesIndexesInLearning)
            {
                double f = getCosineSimilarity(this._learning[x], node.classifier, node.paramThreshold);
                indicator[cnt] = (byte)((f >= node.threshold) ? 1 : 0);
                ++cnt;
            }

            bool isAllZeros = true;
            for (int i = 0; i < samplesIndexesInLearning.Length; ++i)
            {
                if (indicator[i] != 0)
                {
                    isAllZeros = false;
                    break;
                }
            }
            if (isAllZeros == true)
                indicator[0] = 1;

            bool isAllOnes = true;
            for (int i = 0; i < samplesIndexesInLearning.Length; ++i)
            {
                if (indicator[i] == 0)
                {
                    isAllOnes = false;
                    break;
                }
            }
            if (isAllOnes == true)
                indicator[0] = 0;

            return indicator;
        }
        protected static T[][] ToJaggedArray<T>(T[,] twoDimensionalArray)
        {
            int rowsFirstIndex = twoDimensionalArray.GetLowerBound(0);
            int rowsLastIndex = twoDimensionalArray.GetUpperBound(0);
            int numberOfRows = rowsLastIndex + 1;

            int columnsFirstIndex = twoDimensionalArray.GetLowerBound(1);
            int columnsLastIndex = twoDimensionalArray.GetUpperBound(1);
            int numberOfColumns = columnsLastIndex + 1;

            T[][] jaggedArray = new T[numberOfRows][];
            for (int i = rowsFirstIndex; i <= rowsLastIndex; i++)
            {
                jaggedArray[i] = new T[numberOfColumns];

                for (int j = columnsFirstIndex; j <= columnsLastIndex; j++)
                {
                    jaggedArray[i][j] = twoDimensionalArray[i, j];
                }
            }
            return jaggedArray;
        }
        protected static T[,] To2DArray<T>(T[][] jaggedArray)
        {
            T[,] result = new T[jaggedArray.Length, jaggedArray[0].Length];
            for (int i = 0; i < jaggedArray.Length; i++)
            {
                for (int k = 0; k < jaggedArray[0].Length; k++)
                {
                    result[i, k] = jaggedArray[i][k];
                }
            }
            return result;
        }

        protected virtual void OnTreeCreated(EventArgs e)
        {
            EventHandler handler = _TreeCreated;
            if (handler != null)
                handler(this, e);
        }

        
    }
    public class RandomForestNode
    {
        public RandomForestNode left;
        public RandomForestNode right;
        public double[] classifier;
        public double threshold;
        public String category = null;
        public List<int> paramThreshold = new List<int>();
    }
    public class Prediction
    {
        public String predictedClass;
        public String maxClass;
        public int treePredictedClass;
        public int totalTrees;
    }
    public class PredictionPercentage
    {
        public string categorie;
        public int predicted;
        public int total;

        public double Percentage
        {
            get
            {
                if (total != 0)
                    return 100.0 * (double)predicted / (double)total;
                else return -1;
            }
        }


        public PredictionPercentage(string categorie, int predicted, int total)
        {
            // TODO: Complete member initialization
            this.categorie = categorie;
            this.predicted = predicted;
            this.total = total;
        }
    }
    public class ConfusionData
    {
        public List<List<int>> data = new List<List<int>>();
        protected List<String> uniqueCat = new List<string>();

        public List<PredictionPercentage> pourcentages = new List<PredictionPercentage>();

        public List<String> UniqueCat
        {
            get { return uniqueCat; }
            set
            {
                uniqueCat = value;
                foreach (var c in uniqueCat)
                {
                    pourcentages.Add(new PredictionPercentage(c, 0, 0));
                    List<int> row = new List<int>();
                    foreach (var d in uniqueCat)
                        row.Add(0);
                    data.Add(row);
                }
            }
        }
        protected double percentageError = -1;

        public double PercentageError
        {
            get
            {
                //Si la valeur n'a pas été calculée, on la calcule
                if (this.percentageError == -1)
                    this.percentageError = computeError();
                return this.percentageError;
            }
        }
        //Calcule le pourcentage d'erreur d'une matrice
        public double computeError()
        {
            int sum = 0;
            int error = 0;
            for (int i = 0; i < data.Count; i++)
            {
                for (int j = 0; j < data.Count; j++)
                {
                    //On compte le nombre d'individus dans la matrice
                    sum += data[i][j];
                    //Si on n'est pas sur la diagonale, c'est une erreur.
                    if (i != j)
                        error += data[i][j];
                }
            }
            return 100.0 * error / sum;
        }
    }
}