__kernel void mix_kernel(__write_only image2d_t output,
		__read_only image2d_t input_l,
		__read_only image2d_t input_r,
		unsigned char MixTypeChar)
{
	typedef enum
	{
		E_Add = 0,
		E_Subtract = 1,
		E_Multiply = 2,
		E_Min = 3,
		E_Max = 4
	} EMixType;

	EMixType MixType = (EMixType)MixTypeChar;

	int x = get_global_id(0);
	int y = get_global_id(1);
	int h = get_image_height(output);
	int w = get_image_width(output);
	
	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	float4 valueL = read_imagef(input_l, sampler, (int2)(x, y));
	float4 valueR = read_imagef(input_r, sampler, (int2)(x, y));
	
	switch (MixType)
	{
		case E_Add:
			write_imagef(output, (int2)(x, y), valueL + valueR);
			break;

		case E_Subtract:
			write_imagef(output, (int2)(x, y), valueL - valueR);
			break;

		case E_Multiply:
			write_imagef(output, (int2)(x, y), valueL * valueR);
			break;

		case E_Min:
			write_imagef(output, (int2)(x, y), min(valueL, valueR));
			break;

		case E_Max:
			write_imagef(output, (int2)(x, y), max(valueL, valueR));
			break;

		default:
			write_imagef(output, (int2)(x, y), valueL);
			break;
	}
}