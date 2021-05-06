using Microsoft.ML.Data;

namespace BertOnnx
{
    public class Result
    {
        [VectorType(1,256,9)]
        [ColumnName("output_0")]
        public float[] Output { get; set; }
    }
}
