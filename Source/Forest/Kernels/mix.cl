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
	uint4 valueL = read_imageui(input_l, sampler, (int2)(x, y));
	uint4 valueR = read_imageui(input_r, sampler, (int2)(x, y));
	
	switch (MixType)
	{
		case E_Add:
			if (valueL.x + valueR.x < USHRT_MAX)
			{
				write_imageui(output, (int2)(x, y), valueL + valueR);
			}
			else
			{
				write_imageui(output, (int2)(x, y), USHRT_MAX);
			}
			break;

		case E_Subtract:
			if ((int)valueL.x - (int)valueR.x > 0)
			{
				write_imageui(output, (int2)(x, y), valueL - valueR);
			}
			else
			{
				write_imageui(output, (int2)(x, y), 0);
			}
			break;

		case E_Multiply:
			write_imageui(output, (int2)(x, y), valueL * valueR);
			break;

		case E_Min:
			write_imageui(output, (int2)(x, y), min(valueL, valueR));
			break;

		case E_Max:
			write_imageui(output, (int2)(x, y), max(valueL, valueR));
			break;

		default:
			write_imageui(output, (int2)(x, y), valueL);
			break;
	}
}