using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EvalDataStreams1
{
    public class Ejercicio2
    {
        public void Run()
        {
            int acierto = 0;
            const int NumValores = 100;
            const decimal TercerCuartil = (decimal)NumValores * 0.75m;
            const int NIntentos = 1000000;
            var ls = new List<int>();

            for(var i = 0; i < NumValores; i++)
            {
                ls.Add(i);
            }
            for (var i = 0; i < NIntentos; i++)
            {
                Common.Shuffle(ls);
                if(ls[0] >= TercerCuartil || ls[1] >= TercerCuartil || ls[2] >= TercerCuartil || ls[3] >= TercerCuartil || ls[4] >= TercerCuartil)
                {
                    acierto++;
                }
            }
            Console.WriteLine("probabilidad de error " + (decimal)(NIntentos - acierto) / (decimal)NIntentos);
        }
    }
}
