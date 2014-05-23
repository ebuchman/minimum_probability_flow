import numpy as np

def _print_debug(return_values):
		y, y1, wx_b1, e_wx_b1, pairsum1, first1, pairprod1, pairprodWh1, logterm1, hidden_term1, vbias_term1, energy1,\
		wx_b2, e_wx_b2, pairsum2, first2, pairprod2, pairprodWh2, logterm2, hidden_term2, vbias_term2, energy2, \
		z, kcost = return_values

		print '########################## DEBUG OUTPUT ################################'
		print ' y ################################################'
		print y
		print ' y1 ################################################'
		print y1
		print ' wx_b1 ################################################'
		print wx_b1
		print ' e_wx_b1 ################################################'
		print e_wx_b1
		print ' pairsum1 ################################################'
		print pairsum1
		print ' first1 ################################################'
		print first1
		print ' pairprod1 ################################################'
		print pairprod1
		print ' pairprodWh1  ################################################'
		print pairprodWh1
		print ' logterm1 ################################################'
		print logterm1
		print ' hidden_term1 ################################################'
		print hidden_term1
		print ' vbias_term1 ################################################'
		print vbias_term1
		print ' energy1 ################################################'
		print energy1


		print ' wx_b2 ################################################'
		print wx_b2
		print ' e_wx_b2 ################################################'
		print e_wx_b2
		print ' pairsum2 ################################################'
		print pairsum2
		print ' first2 ################################################'
		print first2
		print ' pairprod2 ################################################'
		print pairprod2
		print ' pairprodWh2  ################################################'
		print pairprodWh2
		print ' logterm2 ################################################'
		print logterm2
		print ' hidden_term2 ################################################'
		print hidden_term2
		print ' vbias_term2 ################################################'
		print vbias_term2
		print ' energy2 ################################################'
		print energy2

		print ' z ################################################'
		print z
		print ' kcost ################################################'
		print kcost


def print_debug(return_values, last_debug):
	y, y1, wx_b1, e_wx_b1, pairsum1, first1, pairprod1, pairprodWh1, logterm1, hidden_term1, vbias_term1, energy1,\
	wx_b2, e_wx_b2, pairsum2, first2, pairprod2, pairprodWh2, logterm2, hidden_term2, vbias_term2, energy2, \
	z, kcost = return_values


	if np.isnan(z).sum() or np.isnan(kcost):
		_print_debug(last_debug)
		_print_debug(return_values)
		print 'y', np.isnan(y).sum()
		print 'y1', np.isnan(y1).sum()
		print 'wx_b1', np.isnan(wx_b1).sum()
		print 'e_wx_b1', np.isnan(e_wx_b1).sum()
		print 'pairsum1', np.isnan(pairsum1).sum()
		print 'first1', np.isnan(first1).sum()
		print 'pairprod1', np.isnan(pairprod1).sum()
		print 'pairprodWh1', np.isnan(pairprodWh1).sum()
		print 'logterm1', np.isnan(logterm1).sum()
		print 'hiddenterm1', np.isnan(hidden_term1).sum()
		print 'vbias_term1', np.isnan(vbias_term1).sum()
		print 'energy1', np.isnan(energy1).sum()
		print 'wx_b2', np.isnan(wx_b2).sum()
		print 'e_wx_b2', np.isnan(e_wx_b2).sum()
		print 'pairsum2', np.isnan(pairsum2).sum()
		print 'first2', np.isnan(first2).sum()
		print 'pairprod2', np.isnan(pairprod2).sum()
		print 'pairprodWh2', np.isnan(pairprodWh2).sum()
		print 'logterm2', np.isnan(logterm2).sum()
		print 'hiddenterm2', np.isnan(hidden_term2).sum()
		print 'vbias_term2', np.isnan(vbias_term2).sum()
		print 'energy2', np.isnan(energy2).sum()
		print 'z', np.isnan(z).sum()
		print 'kcost', np.isnan(kcost)
		quit()
