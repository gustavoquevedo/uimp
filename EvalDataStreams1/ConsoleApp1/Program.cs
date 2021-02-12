using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace EvalDataStreams1
{
    class Program
    {
        private const string Filename = "numbers.txt";
        private static Random rnd1 = new Random();
        private static Random rnd2 = new Random();
        private static Stopwatch sw = new Stopwatch();

        private static int N = 1000;

        static void Main(string[] args)
        {
            var sequence = GenerateSequence();
            WriteRandomSequenceFile(sequence);
            var numbers = ReadFile();

            GetMissingNumber(numbers, GFG.GetMissingNo1);
            GetMissingNumber(numbers, GFG.GetMissingNo2);
            GetMissingNumber(numbers, GFG.GetMissingNo3);
            GetMissingNumber(numbers, GFG.GetMissingNo4);
        }

        private static void GetMissingNumber(int[] numbers, Func<int[], int, int> getMissingNumber)
        {
            sw.Restart();
            var missingNumber = getMissingNumber(numbers, N);
            sw.Stop();
            Console.WriteLine("Número={0}, Tiempo={1}", missingNumber, sw.Elapsed);
        }

        private static IEnumerable<int> GenerateSequence()
        {
            var missingNumber = rnd2.Next(N);
            var intArray = new int[N - 1];
            var counter = 0;
            for (int i = 0; i < N; i++)
            {
                if (missingNumber != i)
                {
                    intArray[counter++] = i;
                }
            }
            return Shuffle(intArray.ToList());
        }

        public static IEnumerable<T> Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rnd1.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
            return list;
        }

        private static void WriteRandomSequenceFile(IEnumerable<int> sequence)
        {
            try
            {
                //Pass the filepath and filename to the StreamWriter Constructor
                var sw = new StreamWriter(Filename);

                foreach (var i in sequence)
                {
                    sw.WriteLine(i);
                }
                //Close the file
                sw.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            finally
            {
                Console.WriteLine("Executing finally block.");
            }
        }

        private static int[] ReadFile()
        {
            var numbers = new List<int>();
            //Pass the file path and file name to the StreamReader constructor
            var sr = new StreamReader(Filename);
            //Read the first line of text
            var line = sr.ReadLine();
            //Continue to read until you reach end of file
            while (line != null)
            {
                numbers.Add(int.Parse(sr.ReadLine()));
            }
            //close the file
            sr.Close();
            return numbers.ToArray();
        }
    }
}
